import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):

    def __init__(self, requires_grad=True):
        super(GraphAttentionLayer, self).__init__()
        
        
        if requires_grad:
            self.beta = Parameter(torch.Tensor(1).uniform_(
                0, 1), requires_grad=requires_grad).cuda()
            self.bias = Parameter(torch.Tensor([1e-7]), requires_grad=requires_grad).cuda()
        else:
            self.beta = Variable(torch.zeros(1), requires_grad=requires_grad).cuda()
            self.bias = Variable(torch.Tensor([1e-7]), requires_grad=requires_grad).cuda()
            

    def forward(self, x, adj):
        norm2 = torch.norm(x, 2, 2).view(x.shape[0], -1, 1)
        
        cos = self.beta * torch.div(torch.bmm(x, x.permute(0,2,1)), torch.bmm(norm2, norm2.permute(0, 2, 1)) + self.bias)
        mask = (1. - adj) * -1e9
        masked = cos + mask
        P = F.softmax(masked, dim=1)

        output = torch.bmm(P, x)
        return output


class AGNN(nn.Module):

    def __init__(self, nfeat, nhid, nclass, nlayers, dropout_rate):
        super(AGNN, self).__init__()

        self.layers = nlayers
        self.dropout_rate = dropout_rate
        self.embeddinglayer = nn.Linear(nfeat, nhid)
        nn.init.xavier_uniform(self.embeddinglayer.weight)

        self.attentionlayers = nn.ModuleList()
        self.attentionlayers.append(GraphAttentionLayer(requires_grad=False))
        for i in range(1, self.layers):
            self.attentionlayers.append(GraphAttentionLayer())

        self.outputlayer = nn.Linear(nhid, nclass)
        self.drop_out = nn.Dropout(self.dropout_rate, inplace=False) 
        self.relu = nn.ReLU()
        nn.init.xavier_uniform(self.outputlayer.weight)
        self.log_softmax = nn.LogSoftmax(dim=2)

    def forward(self, data):
        x, adj = data
        x = self.drop_out(self.relu(self.embeddinglayer(x)))

        for i in range(self.layers):
            x = self.attentionlayers[i](x, adj)

        x = self.outputlayer(x)
        x = self.drop_out(x)
        return self.log_softmax(x), x