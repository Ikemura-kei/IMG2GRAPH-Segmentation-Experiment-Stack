import torch
import torch.nn.functional as F
from torch_geometric.nn.conv import GINConv
from torch_geometric.data import Data, Batch
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN
from torch_geometric.nn import GINConv, global_mean_pool, JumpingKnowledge

class GIN(torch.nn.Module):
    def __init__(self, in_channels, num_layers, hidden, num_classes, drop_out=0.2):
        super(GIN, self).__init__()
        self.conv1 = GINConv(
            Sequential(
                Linear(in_channels, hidden),
                ReLU(),
                Linear(hidden, hidden),
                ReLU(),
                BN(hidden),
            ), train_eps=True)
        self.convs = torch.nn.ModuleList()
        self.dropout = drop_out
        for i in range(num_layers - 1):
            self.convs.append(
                GINConv(
                    Sequential(
                        Linear(hidden, hidden),
                        ReLU(),
                        Linear(hidden, hidden),
                        ReLU(),
                        BN(hidden),
                    ), train_eps=True))
        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, num_classes)
        
    def adj_to_cora(self, A):
        # from b, n, n to b, 2, edge
        b = A.shape[0]
        e_list = []
        for i in range(b):
            e = (A[i] != 0).nonzero(as_tuple=False)
            e_list.append(e.t())
        return e_list
        
    def forward(self, data):
        x, A = data
        b = x.shape[0]
        n = x.shape[1]
        data_list = []
        elist = self.adj_to_cora(A)
        for i in range(b):
            data_list.append(Data(x=x[i], edge_index=elist[i]))
        batch = Batch.from_data_list(data_list)
        
        x = batch.x
        edge_index = batch.edge_index
        
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        x = x.view(b, n, -1)
        return F.log_softmax(x, dim=-1), x