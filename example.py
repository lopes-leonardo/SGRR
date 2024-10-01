from sgrr.reranking import Sgrr
from sgrr.utils import compute_map
import numpy as np
import math

# Parameters
K = 80
T = 2
P = 0.5
VERBOSE = True

# Load features and classes
features = np.load(f"./data/flowers-resnet.npy", allow_pickle=True)
classes = np.asarray([math.floor(i / 80) for i in range(len(features))])

# Run SGRR
reranker = Sgrr(K, T, P, verbose=VERBOSE)
reranker.rerank(features, max(classes) + 1)
print("Embeddings Shape:", reranker.embeddings.shape)
print("Output Ranked Lists Shape:", reranker.ranked_lists.shape)
print("Output Ranked Lists MAP:", compute_map(reranker.ranked_lists, classes))
