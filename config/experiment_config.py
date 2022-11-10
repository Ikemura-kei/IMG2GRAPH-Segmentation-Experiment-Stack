import torch
import numpy as np

class ExperimentConfig(object):
    graph_gen_net = 'scg-net'

    root_dir = "/"
    dataset_dir = "dataset"
    train_data_dir = "train/"
    val_data_dir = "test/"

    # data set parameters
    nb_classes = 894

    weights = []

    train_img_size = [64, 64]
    val_img_size = [64, 64]

    train_batch = 2
    val_batch = 1

    seeds = 69278 # random seed

    # training hyper parameters
    lr = 1.5e-4 / np.sqrt(3.0)
    lr_decay = 0.9
    max_iter = 1e8

    # l2 regularization factor, increasing or decreasing to fight over-fitting
    weight_decay = 2e-5
    momentum = 0.9

    # check point parameters
    print_freq = 100
    save_pred = False
    save_rate = 0.1
    best_record = {}
    log_dir = "log"

    model_save_path = ""
    save_freq = 20

    epoch = 500
    experiment_name = "experiment"
    result_img_path = ""
    val_freq = 1

    num_nodes = (10, 10)

    device = "cpu"

    dataset = "cifar_mnist"

    gnn = ""

    def __init__(self, graph_gen_net):
        self.graph_gen_net = graph_gen_net


import yaml

def load_config(config_file):
    with open(config_file, "r") as stream:
        try:
            ret = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return ret