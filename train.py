import os
import time
import sys

import numpy as np

import cv2
import natsort 

from dataloader.dataset import NYUDepthV2Dataset
from dataloader.cifar_mnist import CifarMnist
from model.create_model import SegmentationGNN
from loss.acw_loss import ACW_loss
from optimizer.lookahead import Lookahead
from config.experiment_config import ExperimentConfig
from utils.visualization_utils import visualize_seg_map
from utils.visualization_utils import test_dataset

from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch

def train(model, train_loader, val_loader, optimizer, loss_func, exp_config, lr_scheduler):

    writer = SummaryWriter(log_dir=os.path.join(exp_config.root_dir, exp_config.log_dir, exp_config.experiment_name))

    cur_epoch = 0
    num_iter_per_epoch = len(train_loader)
    start_time = time.time()

    if len(os.listdir(exp_config.model_save_path)) > 0:
        # load previously learned weigths
        ckpt_list = natsort.natsorted(os.listdir(exp_config.model_save_path), reverse=True)

        for ckpt in ckpt_list:
            if ckpt.endswith(".pth"):
                checkpoint_dict = torch.load(os.path.join(exp_config.model_save_path, ckpt))

                cur_epoch = checkpoint_dict["epoch"]
                model.load_state_dict(checkpoint_dict['model_state_dict'])
                optimizer.load_state_dict(checkpoint_dict['optim_state_dict'])

                print("\n#################################################\n### loaded checkpoint! starting from epoch %d ###\n#################################################\n" % (cur_epoch))
                break

    while cur_epoch < exp_config.epoch:
        model.train()

        for i, (inputs, labels) in enumerate(train_loader):

            inputs, labels = inputs.to(exp_config.device), labels.to(exp_config.device).long()

            out, cost = model(inputs)

            loss = loss_func(out, labels)

            optimizer.zero_grad()

            loss_total = loss + cost

            loss_total.backward()

            optimizer.step()

            lr_scheduler.step()

            if i % exp_config.print_freq == 0:
                cur_time = time.time()
                # print log information
                print("[epoch %d], [iter %d / %d], [lr %.5f], [loss %.5f], [time %.3f]" % (cur_epoch+1, i+1, num_iter_per_epoch, optimizer.param_groups[0]['lr'], loss_total.item(), cur_time-start_time))

            writer.add_scalar(tag="train/loss", scalar_value=loss_total.item(), global_step=num_iter_per_epoch*cur_epoch+i)
            writer.add_scalar(tag="lr", scalar_value=optimizer.param_groups[0]['lr'], global_step=num_iter_per_epoch*cur_epoch+i)

        cur_epoch += 1
        
        if cur_epoch % exp_config.save_freq == 0:
            checkpnt_dict = {'epoch': cur_epoch, 'model_state_dict': model.state_dict(), 'optim_state_dict': optimizer.state_dict()}
            torch.save(checkpnt_dict, os.path.join(exp_config.model_save_path, str(cur_epoch)+".pth"))

        model.eval()
        avg_loss = 0
        counter = 0
        with torch.no_grad():
            if cur_epoch % exp_config.val_freq == 0:
                for i, (inputs, labels) in enumerate(val_loader):

                    inputs, labels = inputs.to(exp_config.device), labels.to(exp_config.device).long()

                    out = model(inputs)

                    loss = loss_func(out, labels)
                    avg_loss += loss.item()
                    counter += 1

                    if i == 0: # save the first result for visualization
                        seg_map = np.argmax(out[0].detach().cpu().numpy(), axis=0)
                        save_img = visualize_seg_map(seg_map)
                        cv2.imwrite(os.path.join(exp_config.result_img_path, "pred"+str(cur_epoch)+".jpg"), save_img)

                        seg_map = labels[0].detach().cpu().numpy()
                        # print(seg_map.shape)
                        save_img_2 = visualize_seg_map(seg_map)
                        cv2.imwrite(os.path.join(exp_config.result_img_path, "gt"+str(cur_epoch)+".jpg"), save_img_2)
                        
                print("average validation accuracy:", avg_loss / counter)
                    

def main():
    exp_config = ExperimentConfig(graph_gen_net="scg-net")
    exp_config.train_img_size = (64, 64)
    exp_config.val_img_size = (64, 64)
    exp_config.train_batch = 16
    exp_config.root_dir = "/data1/kikemura/IMG2GRAPH_Segmentation_Experiment"
    exp_config.print_freq = 10
    exp_config.val_freq = 15
    exp_config.save_freq = 25
    exp_config.model_save_path = "/data1/kikemura/IMG2GRAPH_Segmentation_Experiment/saved_weights/" + exp_config.experiment_name
    exp_config.result_img_path = "/data1/kikemura/IMG2GRAPH_Segmentation_Experiment/results/" + exp_config.experiment_name
    exp_config.device = "cuda"
    exp_config.nb_classes = 10

    os.makedirs(exp_config.model_save_path, exist_ok=True) 
    os.makedirs(exp_config.result_img_path, exist_ok=True) 

    transform = transforms.Compose([transforms.Resize(exp_config.train_img_size), transforms.ToTensor()])
    if exp_config.dataset == "cifar_mnist":
        train_data = CifarMnist(os.path.join(exp_config.root_dir, exp_config.dataset_dir, exp_config.dataset, exp_config.train_data_dir), exp_config.train_img_size, transform)
    train_loader = DataLoader(train_data, batch_size=exp_config.train_batch, shuffle=True)

    transform = transforms.Compose([transforms.Resize(exp_config.val_img_size), transforms.ToTensor()])
    if exp_config.dataset == "cifar_mnist":
        val_data = CifarMnist(os.path.join(exp_config.root_dir, exp_config.dataset_dir, exp_config.dataset, exp_config.val_data_dir), exp_config.val_img_size, transform)
    val_loader = DataLoader(val_data, batch_size=exp_config.val_batch, shuffle=True)
    
    test_dataset(train_loader)

    model = SegmentationGNN(graph_gen_net=exp_config.graph_gen_net, num_classes=exp_config.nb_classes, nb_nodes=(50, 50))
    model.to(exp_config.device)

    loss_func = ACW_loss()

    optimizer_params = model.parameters()
    optimizer_base = torch.optim.Adam(optimizer_params, lr=exp_config.lr, weight_decay=exp_config.weight_decay)
    optimizer = Lookahead(optimizer_base, k=6)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=exp_config.max_iter, eta_min=1e-17)

    train(model, train_loader, val_loader, optimizer, loss_func, exp_config, lr_scheduler)

if __name__ == "__main__":
    main()

   