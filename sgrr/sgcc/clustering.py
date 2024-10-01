from lhrr import LhrrWrapper
import numpy as np


class Sgcc:
    def __init__(
        self,
        k: int,
        t: int,
        p=0.5,
        verbose: bool = False,
    ):
        self.k = k
        self.t = t
        self.p = p
        self.verbose = verbose
        self._set_initial_state()

    def _check_print(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def _set_initial_state(self):
        self.scores = None
        self.leaders = None

    def _clusterize(self, c: int) -> None:
        self._check_print("Finding leaders...")
        self.n_clusters = c
        self._find_leaders(self.n_clusters)
        self.clusters = []
        self.labels = np.full(self.n, -1)
        targets = self.scores[: self.limit, 0].astype(int)

        self._check_print("Initializing clusters...")
        for i in self.leaders:
            self.clusters.append(np.asarray([i], dtype=int))
            self.labels[i] = len(self.clusters) - 1

        self._check_print("Creating soft-labels...")
        for i in targets:
            if i not in self.leaders:
                self._classify(i)

    def _process_features(self, features: np.array, metric: str = "euclidean") -> None:
        self.features = features
        self.n = len(features)
        self.limit = int(self.n * self.p)
        self._check_print(f"Running LHRR using {metric} distance.")
        self._run_lhrr(metric)

    def _run_lhrr(self, metric: str = "euclidean"):
        self._check_print("Running LHRR...")
        self.lhrr = LhrrWrapper(self.k, self.t)
        self.lhrr.run(self.features, metric=metric)
        self.hyperedges = self.lhrr.get_hyper_edges()
        self.hyperedges = [np.asarray(i) for i in self.hyperedges]
        self.confid = np.asarray(self.lhrr.get_confid_scores())
        self.ranked_lists = self.lhrr.get_ranked_lists()

    def _find_leaders(self, k=int) -> None:
        if self.scores is None:
            self._check_print("Computing scores...")
            self._compute_scores()

        leaders = []
        self._compute_he_matrix()
        self.leader_intersection_score = np.zeros((k, self.n))
        leaders.append(int(self.scores[0, 0]))
        for i in range(1, k):
            leaders.append(self._find_next_leader(leaders))
        self.leaders = leaders

    def _compute_scores(self):
        self_scores = self._biuld_self_score_array()
        scores = self.confid * self_scores
        scores = [[idx, score] for idx, score in enumerate(scores)]
        scores.sort(key=lambda x: x[0])
        self._sorted_scores = np.asarray(scores)
        scores.sort(key=lambda x: x[1])
        scores.reverse()
        self.scores = np.asarray(scores)

    def _biuld_self_score_array(self):
        self_score = []
        for idx, he in enumerate(self.hyperedges):
            pos = np.where(he[:, 0] == idx)[0]
            self_score.append(he[pos][0][1])
        return np.asarray(self_score)

    def _compute_he_matrix(self):
        he_matrix = np.zeros((self.n, self.n))
        for i in range(self.n):
            he = self.hyperedges[i]
            idxs = he[:, 0].astype(int)
            scores = he[:, 1]
            he_matrix[i][idxs] = scores
        self.he_matrix = he_matrix

    def _find_next_leader(self, leaders):
        target = len(leaders) - 1
        target_div = np.dot(self.he_matrix[leaders[target]], self.he_matrix.T)
        self.leader_intersection_score[target] = target_div
        scores = self._sorted_scores[:, 1] / (
            1 + np.sum(self.leader_intersection_score[: target + 1], axis=0)
        )
        selected = np.argsort(scores)
        return int(selected[-1])

    def _classify(self, item: int):
        scores = []
        for leader in self.leaders:
            scores.append(self._classification_score(item, leader))
        scores.sort(key=lambda x: x[1])
        scores.reverse()
        leader = scores[0][0]

        target = self.labels[leader]
        self.clusters[target] = np.append(self.clusters[target], item)
        self.labels[item] = target

    def _classification_score(self, item: int, leader: int):
        cluster = self.clusters[self.labels[leader]]
        s = np.sum(self.he_matrix[cluster, item])
        return [leader, (s / len(cluster))]

    def run(
        self,
        features: np.ndarray,
        c: int,
        metric: str = "euclidean",
    ) -> None:
        self._set_initial_state()
        self._process_features(features, metric)
        self._clusterize(c)
