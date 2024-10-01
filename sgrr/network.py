import torch.nn.functional as F
from torch.nn import Module
from torch_geometric.nn import SGConv


class SGCNet(Module):
    def __init__(self, features, neurons, classes):
        super(SGCNet, self).__init__()
        self.conv1 = SGConv(features, neurons, K=2)
        self.conv2 = SGConv(neurons, classes, K=2)
        self.embedding = None

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        self.embedding = x
        return F.log_softmax(x, dim=1)


def load_sgc(features: int, neurons: int, classes: int):
    return SGCNet(features, neurons, classes)
