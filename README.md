# CMoDE: Adaptive  Semantic  Segmentation in  Adverse  Environmental  Conditions
CMoDE is a deep learning fusion scheme for multimodal semantic image segmentation, where the goal is to assign semantic labels (e.g., car, road, tree and so on) to every pixel in the input image. CMoDE is easily trainable on a single GPU with 12 GB of memory and has a fast inference time. CMoDE is benchmarked on Cityscapes, Synthia, ScanNet, SUN RGB-D and Freiburg Forest datasets.

This repository contains our TensorFlow implementation of CMoDE which allows you to train your own model on any dataset and evaluate the results in terms of the mean IoU metric. 

If you find the code useful for your research, please consider citing our paper:
```
@inproceedings{valada2017icra,
  author = {Valada, Abhinav and Vertens, Johan and Dhall, Ankit and Burgard, Wolfram},
  title = {AdapNet: Adaptive Semantic Segmentation in Adverse Environmental Conditions},
  booktitle = {Proceedings of the IEEE International Conference on Robotics and Automation (ICRA)},
  pages={4644--4651},
  year = {2017},
  organization={IEEE}
}
```
```
@inproceedings{valada2016irosws,
  author={Valada, Abhinav and Dhall, Ankit and Burgard, Wolfram},
  title={Convoluted Mixture of Deep Experts for Robust Semantic Segmentation},
  booktitle={IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) Workshop, State Estimation and Terrain Perception for All Terrain Mobile Robots},
  year={2016}
}
```
## Live Demo
http://deepscene.cs.uni-freiburg.de

## Example Segmentation Results

| Dataset       | Modality 1     |Modality 2    | Segmented Image|
| ------------- | -------------- |------------- |--------------- |
| Cityscapes    |<img src="images/city.png" width=200> | <img src="images/city_jet.png" width=200> | <img src="images/city_pred.png" width=200>|
| Forest  | <img src="images/forest.png" width=200>  | <img src="images/forest_evi.png" width=200>  |<img src="images/forest_prediction.png" width=200> |
| Sun RGB-D  | <img src="images/sun.png" width=200>  |<img src="images/sun_hha.png" width=200>  | <img src="images/sun_pred.png" width=200>|
| Synthia  | <img src="images/synthia.png" width=200>  |<img src="images/synthia_jet.png" width=200>  | <img src="images/synthia_pred.png" width=200> |
| ScanNet v2  | <img src="images/scannet.png" width=200>  |<img src="images/scannet_hha.png" width=200>  |<img src="images/scannet_pred.png" width=200> |

## Contacts
* [Abhinav Valada](http://www2.informatik.uni-freiburg.de/~valada/)
* [Rohit Mohan](https://github.com/mohan1914)

## System Requirements

#### Programming Language
```
Python 2.7
```
#### Python Packages
```
tensorflow-gpu 1.4.0
```
## Configure the Network

First train individual [AdapNet](https://github.com/DeepSceneSeg/AdapNet) or [AdapNet++](https://github.com/DeepSceneSeg/AdapNet-pp) model for modality 1 and modality 2 in the dataset. We will use this pre-trained modality-secific models for initializing our CMoDE network.

#### Data

* Augment the training data.
  In our work, we first resized the images in the dataset to 768x384 pixels and then apply a series of augmentations (random_flip, random_scale and random_crop). The image corresonding to each modality and the label should be augmented together using the same parameters.

* Convert the training data (augmented), test data and validation data into the .tfrecords format.
  Create a .txt file for each set having entries in the following format:
  ```
     path_to_modality1/0.png path_to_modality2/0.png path_to_label/0.png
     path_to_modality1/1.png path_to_modality2/1.png path_to_label/1.png
     path_to_modality1/2.png path_to_modality2/2.png path_to_label/2.png
     ...
  ```
 Run the convert_to_tfrecords.py from dataset folder for each of the train, test, val sets to create the tfrecords:
  ```
     python convert_to_tfrecords.py --file path_to_.txt_file --record tf_records_name 
  ```
  (Input to model is in BGR and 'NHWC' form)

#### Training
```
    gpu_id: id of gpu to be used
    model: name of the model
    num_classes: number of classes
    checkpoint1:  path to pre-trained model for modality 1 (rgb)
    checkpoint2:  path to pre-trained model for modality 2 (jet,hha,evi)
    checkpoint: path to save model
    train_data: path to dataset .tfrecords
    batch_size: training batch size
    skip_step: how many steps to print loss 
    height: height of input image
    width: width of input image
    max_iteration: how many iterations to train
    learning_rate: initial learning rate
    save_step: how many steps to save the model
    power: parameter for poly learning rate
    
```
#### Evaluation
```
    gpu_id: id of gpu to be used
    model: name of the model
    num_classes: number of classes
    checkpoint: path to saved model
    test_data: path to dataset .tfrecords
    batch_size: evaluation batch size
    skip_step: how many steps to print mIoU
    height: height of input image
    width: width of input image
    
```

#### Please refer our [paper](https://arxiv.org/pdf/1808.03833.pdf) for the dataset preparation procedure for each modality and the training protocol to be employed.

## Training and Evaluation

#### Training
Edit the config file for training in config folder.
Run:
```
python train.py -c config cityscapes_train.config or python train.py --config cityscapes_train.config

```

#### Evaluation

Select a checkpoint to test/validate your model in terms of the mean IoU.
Create the config file for evaluation in the config folder.

```
python evaluate.py -c config cityscapes_test.config or python evaluate.py --config cityscapes_test.config
```

## Additional Notes:
   * We provide the CMoDE fusion implementation for either AdapNet or AdapNet++ as the expert network architecture. You can swap the expert network with any architecture of your choosing by modifying models/CMoDE.py script.
   * We only provide the single scale evaluation script. Multi-Scale+Flip evaluation further imporves the performance of the model.
   * The code in this repository only performs training on a single GPU. Multi-GPU training using synchronized batch normalization with larger batch size further improves the performance of the model.
   * Initializing the model with pre-trained weights from large datasets such as the Mapillary Vistas and BDD100K yields an improved performance.
   
## License
For academic usage, the code is released under the [GPLv3](https://www.gnu.org/licenses/gpl-3.0.en.html) license. For any commercial purpose, please contact the authors.
