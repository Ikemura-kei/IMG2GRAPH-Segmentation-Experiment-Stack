# IMG2GRAPH Segmentation Experiment Stack

This repository provide various models and experiments for testing various graph generation methods from images. The generated graph is then passed along to a various GNN models for message passing. The updated graph is subsequently converted to segmentation maps.

# Dependencies
- pytorch 1.11.0
- python 3.9.12

# Set up

## Environment
After installing the dependencies with correct version mentioned in Dependencies section, run the following:

```
pip install -r requirements.txt
```

## Dataset
We use the following list of custom/public dataset for benchmarking:
* cifar-mnist: custom dataset that overlays mnist digits onto cifar-10 images for segmentation (i.e. segmenting the digits)
* nyu-depthv2-mini: randomly selected 20% of publically available NYU-DepthV2 dataset

The download links for these dataset are as follows:
* cifar-mnist:
* nyu-depthv2-mini:

The dataset structure shall be:
- dataset
  - cifar_mnist
    - test
      - rgb
      - seg_map
    - train
      - rgb
      - seg_map

# Train
## Configuration
You will need to put your configurations in `config/config.yaml`

- batch_size: should be an integer being at least 1, be careful of GPU RAM overflow if this is set too large
- dataset: one of __{"cifar_mnist", "nyu_depthv2_mini"}__
- root_dir: your project root dir, should be .../IMG2GRAPH_Segmentation_Experiment_Stack where ... is where you put this project into.
- experiment_name: the unique identifier you give to everything saved for this experiment, folders named by this will be created for different saving purposes.
- model_save_dir: a folder to save your checkpoints. They will be saved under `root_dir/model_save_dir/experiment_name/`
- save_pred: boolean value, `True` for saving predictions and groudtruths for evey validation, `False` to disable.
- result_save_dir: a folder to save your groundtruth and prediction pairs during training. They will be saved under `root_dir/result_save_dir/experiment_name/`, will be ignored if `save_pred` is `False`.
- val_image_size: the size of image used for validation, should be at least (32, 32). A recommended size is (64, 64).
- train_image_size: the size of image used for training, should be at least (32, 32). A recommended size is (64, 64).
- graph_gen_model: the model for generating graphs out of images, should be one of __{"vanilla", "scg-net", "graph-fcn"}__
- print_freq: number of epochs to print training status, including loss, learning rate, etc.
- save_freq: number of epochs to save a checkpoint.
- val_freq: number of epochs to perform validation, results will be saved to `root_dir/result_save_dir/experiment_name/`.
- device: if you somehow happens to prefer using CPUs (though not recommended), you can set here.
- num_nodes: the number of nodes in the graph, this will affect the resolution of the prediction as we model each node as a pixel.
- learning_rate: training learning rate
- gnn: the type of gnn layer used, should be one of __{"gcn", "cheb", "sage"}__

## Run
After making sure the configurations are what you want, run the following command

```
python train.py
```

> Note: currently only support single GPU training, we will make it distributed over multiple GPU in the future.


# Adding New Models
You can feel free to add any segmentation networks that incorporates graphs generated from images and/or new GNN layers for more variety in choices of experiment components.
