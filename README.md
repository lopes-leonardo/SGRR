# Self-Supervised Image Re-Ranking based on Hypergraphs and Graph Convolutional Networks

This repository contains the implementation of  **Self-Supervised GCN for Re-Ranking (SGRR)**.
The paper abstract can be seen below:

> Image retrieval approaches typically involve two fundamental stages: visual content representation and similarity measurement. Traditional methods rely on pairwise dissimilarity metrics, such as Euclidean distance, which overlook the global structure of datasets. Aiming to address this limitation, various unsupervised post-processing approaches have been developed to redefine similarity measures. Diffusion processes and rank-based methods compute a more effective similarity by considering the relationships among images and the overall dataset structure. However, neither approach is capable of defining novel image representations. This paper aims to overcome this limitation by proposing a novel self-supervised image re-ranking method. The proposed method exploits a hypergraph model, clustering strategies, and Graph Convolutional Networks (GCNs). Initially, an unsupervised rank-based manifold learning method computes global similarities to define small and reliable clusters, which are used as soft labels for training a semi-supervised GCN model. This GCN undergoes a two-stage training process: an initial classification-focused stage followed by a retrieval-focused stage. The final GCN embeddings are employed for retrieval tasks using the cosine similarity. An experimental evaluation conducted on four public datasets with three different visual features indicates that the proposed approach outperforms traditional and recent rank-based methods.

## Instalatioin

To run SGRR, please make sure [docker](https://docs.docker.com/engine/install/), [NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) and a proper GPU are configured in your host machine.

Afterwards, build our preconfigured docker image using the following command inside the project's root folder:

```shell
docker build -t sgrr_image:latest .
```

After building, run a new container using the following code:

```shell
docker run --gpus all --name sgrr_container -it sgrr_image:latest
```

Inside the docker image, you can run our [example code](./example.py) or any of other reranking script:

```shell
python example.py
```

## Usage

`SGRR` main class can be imported, instanciated, and used as follows:

```python
from sgrr.reranking import Sgrr
reranker = Sgrr(K, T, P)
reranker.rerank(features, number_of_classes)
```

After executing, both new embeddings and the output ranked lists will be available in the class instance:
```python
print("Embeddings Shape:", reranker.embeddings.shape)
print("Output Ranked Lists Shape:", reranker.ranked_lists.shape)
```