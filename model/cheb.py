import torch
import torch.nn.functional as F
from torch_geometric.nn.conv import ChebConv
from torch_geometric.data import Data, Batch

class TwoLayerCheb(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, k=3, dropout=0.2):
        super(TwoLayerCheb, self).__init__()
        
        self.cheb_conv1 = ChebConv(in_channels, hidden_channels, K=k)
        self.cheb_conv2 = ChebConv(hidden_channels, out_channels, K=k)
        
        self.dropout = dropout
        
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
            
        x = self.cheb_conv1(x=batch.x, edge_index=batch.edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.cheb_conv2(x, batch.edge_index)
        x = x.view(b, n, -1)
        
        return x.log_softmax(dim=-1), x