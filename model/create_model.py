import torch.nn as nn
import torch
from .scg_net import *
from .gcn_layer import *
from .vanilla import *

class SegmentationGNN(nn.Module):

    def __init__(self, graph_gen_net, num_classes, nb_nodes, dropout=0.2):
        # constant definition, when new graph generation networks are added, update the definition here
        AVAILABLE_GRAPH_GEN_NETS = ["scg-net", 'graph-fcn', "3d-graph", "vanilla"]
        super(SegmentationGNN, self).__init__()

        assert graph_gen_net in AVAILABLE_GRAPH_GEN_NETS, "parameter 'graph_gen_net' has to be one of {} but got {}".format(AVAILABLE_GRAPH_GEN_NETS, graph_gen_net)

        self.graph_gen_net_name = graph_gen_net

        if graph_gen_net == AVAILABLE_GRAPH_GEN_NETS[0]: # scg-net
            print("using", AVAILABLE_GRAPH_GEN_NETS[0])
            self.graph_gen_net = SCG_Net(nb_nodes, num_classes)

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