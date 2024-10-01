import itertools
import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import Data
from sgcc.clustering import Sgcc
from network import load_sgc


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

    def rerank(
        self,
        features: np.ndarray,
        number_classes: int,
        metric: str = "euclidean",
    ) -> None:
        self.sgcc.run(feature, number_classes, metric)
        self._make_triplets()
        self._train_gcn()
        self._compute_ranked_lists()
