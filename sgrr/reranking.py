import itertools
import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import Data
from .sgcc.clustering import Sgcc
from .network import load_sgc
from .utils import compute_ranked_lists


class Sgrr:

    def __init__(
        self,
        k: int,
        t: int = 2,
        p: float = 0.5,
        nll_epochs: int = 300,
        triplet_epochs: int = 300,
        learning_rate: float = 0.001,
        verbose: bool = False,
    ):
        self.k = k
        self.nll_epochs = nll_epochs
        self.triplet_epochs = triplet_epochs
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.sgcc = Sgcc(k, t, p, verbose)
        self.neurons = 32

    def _check_print(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def _make_triplets(self):
        triplets = np.empty((0, 3))
        all_items = set([i for i in range(self.sgcc.n)])
        for c_idx, c in enumerate(self.sgcc.clusters):
            base_triplets = np.array(list(itertools.permutations(c, 2)))
            base_number = base_triplets.shape[0]
            if base_number == 0:
                continue
            chosen_positives = set(c)
            available_negatives = all_items - chosen_positives
            negatives = np.random.choice(list(available_negatives), size=base_number)
            negatives = np.expand_dims(negatives, axis=1)
            temp_triplets = np.append(base_triplets, negatives, axis=1)
            triplets = np.append(triplets, temp_triplets, axis=0)
        self._check_print(f"Created triplets shape: {triplets.shape}")
        self.triplets = triplets

    def _is_reciprocal(self, anchor: int, target: int, k: int):
        for i in range(k):
            if self.sgcc.ranked_lists[anchor][i] == target:
                return True
        return False

    def _compute_edge_index(self):
        edge_list = []
        target_rank = min(self.k + 1, len(self.sgcc.ranked_lists[0]))
        for i in range(self.sgcc.n):
            rank = self.sgcc.ranked_lists[i]
            # Loop over k+1 positions, since the item itself is one of them
            for j in range(target_rank):
                if (rank[j] != i) and self._is_reciprocal(rank[j], i, target_rank):
                    target = rank[j]
                    edge_list.append((i, target))

        return np.asarray(edge_list).transpose()

    def _nll_loss(self, model_output, data):
        return F.nll_loss(model_output[data.train_mask], data.y[data.train_mask])

    def _triplet_loss(self, model_output, data):
        return F.triplet_margin_loss(
            model_output[self.triplets[:, 0]],
            model_output[self.triplets[:, 1]],
            model_output[self.triplets[:, 2]],
            margin=1,
        )

    def _train_model(
        self,
        model,
        optimizer,
        loss_wrapper,
        num_epochs,
        data,
        debug: bool = True,
    ) -> None:
        model.train()
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            out = model(data)
            loss = loss_wrapper(out, data)
            if debug and (epoch + 1) % 100 == 0:
                self._check_print(
                    f"-> It {epoch+1} / LR: {optimizer.param_groups[0]['lr']} / loss {loss_wrapper.__name__}: {loss.item()}"
                )
            loss.backward()
            optimizer.step()

    def _train_gcn(self) -> None:
        # Device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Data definition
        y = torch.tensor(self.sgcc.labels).to(device)
        x = torch.tensor(self.sgcc.features).to(device)

        # Mask definition
        train_mask = torch.tensor(self.sgcc.labels != -1).to(device)
        val_mask = torch.tensor([]).to(device)
        test_mask = torch.tensor([]).to(device)

        # Edge index
        edge_index = self._compute_edge_index()
        edge_index = torch.tensor(edge_index)
        edge_index = edge_index.contiguous().to(device)
        self.edge_index = edge_index

        # Tensor data
        data = Data(
            x=x.float(),
            edge_index=edge_index,
            y=y,
            test_mask=test_mask,
            train_mask=train_mask,
            val_mask=val_mask,
        )

        # Variables
        feature_dim = len(self.sgcc.features[0])
        model = load_sgc(feature_dim, self.neurons, self.number_classes).to(device)

        # Two-stage GCN training
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.learning_rate, weight_decay=5e-4
        )
        self._train_model(model, optimizer, self._nll_loss, self.nll_epochs, data)
        self._train_model(
            model, optimizer, self._triplet_loss, self.triplet_epochs, data
        )

        model.eval()
        _, pred = model(data).max(dim=1)

        self.embeddings = model.embedding.cpu().detach().numpy()

    def rerank(
        self,
        features: np.ndarray,
        number_classes: int,
        metric: str = "euclidean",
    ) -> None:
        self.number_classes = number_classes
        self.sgcc.run(features, number_classes, metric)
        self._make_triplets()
        self._train_gcn()
        self.ranked_lists = compute_ranked_lists(self.embeddings)
