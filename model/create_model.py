import torch.nn as nn
import torch
from .scg_net import *
from .gcn_layer import *
from .vanilla import *
from .sage import *
from .cheb import *
from .agnn import *
from .gin import *

class SegmentationGNN(nn.Module):

    def __init__(self, graph_gen_net, num_classes, nb_nodes, gnn, dropout=0.2):
        # constant definition, when new graph generation networks are added, update the definition here
        AVAILABLE_GRAPH_GEN_NETS = ["scg-net", 'graph-fcn', "3d-graph", "vanilla"]
        AVAILABLE_GNN = ["gcn", 'sage', "cheb", "agnn", "gin"]

        super(SegmentationGNN, self).__init__()

        assert graph_gen_net in AVAILABLE_GRAPH_GEN_NETS, "parameter 'graph_gen_net' has to be one of {} but got {}".format(AVAILABLE_GRAPH_GEN_NETS, graph_gen_net)

        self.graph_gen_net_name = graph_gen_net

        if gnn == AVAILABLE_GNN[0] or gnn == "": # gcn
            print("using", AVAILABLE_GNN[0])
            self.graph_layers1 = GCN_Layer(1024, out_features=128, bnorm=True, activation=nn.ReLU(True), dropout=dropout)
            self.graph_layers2 = GCN_Layer(128, out_features=num_classes, bnorm=False, activation=None)
            self.gnn = nn.Sequential(self.graph_layers1, self.graph_layers2)
        
        if gnn == AVAILABLE_GNN[1]: # sage
            print("using", AVAILABLE_GNN[1])
            self.gnn = TwoLayerSAGE(1024, 16, num_classes)

        if gnn == AVAILABLE_GNN[2]: # cheb
            print("using", AVAILABLE_GNN[2])
            self.gnn = TwoLayerCheb(1024, 16, num_classes)

        if gnn == AVAILABLE_GNN[3]: # agnn
            print("using", AVAILABLE_GNN[3])
            self.gnn = AGNN(1024, 16, num_classes, 3, 0.5)
        
        if gnn == AVAILABLE_GNN[4]: # gin
            print("using", AVAILABLE_GNN[4])
            self.gnn = GIN(1024, 2, 16, num_classes)

        if graph_gen_net == AVAILABLE_GRAPH_GEN_NETS[0] or graph_gen_model == "": # scg-net
            print("using", AVAILABLE_GRAPH_GEN_NETS[0])
            self.graph_gen_net = SCG_Net(nb_nodes, num_classes, self.gnn)

        if graph_gen_net == AVAILABLE_GRAPH_GEN_NETS[1]: # graph-fcn
            print("using", AVAILABLE_GRAPH_GEN_NETS[1])
            self.graph_gen_net = None

        if graph_gen_net == AVAILABLE_GRAPH_GEN_NETS[2]: # 3d-graph
            print("using", AVAILABLE_GRAPH_GEN_NETS[2])
            self.graph_gen_net = None
        
        if graph_gen_net == AVAILABLE_GRAPH_GEN_NETS[3]: # vanilla
            print("using", AVAILABLE_GRAPH_GEN_NETS[3])
            self.graph_gen_net = Vanilla()

    def forward(self, x):
        return self.graph_gen_net(x)