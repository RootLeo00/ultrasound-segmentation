# Multiclass Semantic Segmentation of Ultrasound Images
The purpose of this thesis is to implement and analyze a use case of Deep Learning tech-
niques for the semantic segmentation of ultrasound images. The dataset used consists of
hand ultrasound images obtained from the database of the Rizzoli Orthopedic Institute.
The data consists of hand images, where each image has been manually annotated with
three different labels. The three classes are: bone, synoviuml cavity, and tendon.
To train the Deep Learning models, a convolutional neural network (CNN) architecture
of the U-net type has been used. The input for the network is an image with dimensions
224 x 224 x 3, and the output is a 3-dimensional vector corresponding to the predicted class
probabilities for each of the three classes. The trained model was then evaluated on a test
set of 60 images. The results showed that the network was able to achieve an accuracy of
over 90.0% on both the training and test sets. Furthermore, since the dataset is imbalanced,
other prediction evaluation metrics besides accuracy, such as the F1-score, were necessary,
achieving an average value greater than 90.0%.


# Requirements
- [Tensorflow 2.x](https://www.tensorflow.org/) (An open source deep learning platform)
- [Keras](https://keras.io/) (An open source deep learning framework)
- [Python 3.x](https://www.python.org/) (Main language to work with Tensorflow) 
- [scikit-learn](https://scikit-learn.org/stable/) (Machine Learning library in Python)
- [numpy](https://numpy.org/) (A fundamental package for scientific computing with Python)
- [Matplotlib](https://matplotlib.org/) (Library for creating static, animated, and interactive visualizations in Python)
- [Argparse](https://docs.python.org/3/library/argparse.html) (Parser for command-line options, arguments and sub-commands in Python)


# Table Of Contents
- [Multiclass Semantic Segmentation of Ultrasound Images](#multiclass-semantic-segmentation-of-ultrasound-images)
- [Requirements](#requirements)
- [Table Of Contents](#table-of-contents)
- [About the project](#about-the-project)
- [In Details](#in-details)
- [Contributing](#contributing)


# About the project 
The software can be used via a command line interface. The interface was created with the help of _argparse_, a Python library with which the developer
defines a program and the required arguments and then argparse helps to parse them from _sys.argv_, i.e. the list of command line arguments. 
To run the application you need to run the command:
```
python3 main.py
``` 
The user interface consists of two programs:
• **Train and predict** which allows you to train a neural network and run a test of predictions
• **Only predict** which allows you to run a predictions test on an already trained neural network
Various parameters or options can be specified for each of the programs. You can see a list of them running:
```
python3 main.py -TP --help
``` 
Every dataset consists of 60 images featuring sagittal sections of the metacarpus, made by a Dr. at the Rizzoli Orthopedic Institute in Bologna.
The images were manually labeled by the author. 
Down below there is an image showing the 3 different configurations (from the top, tendon-bone, tendon-synovyal, synovial-bone):
![alt text](https://github.com/RootLeo00/ultrasound_ML/blob/master/dataset_img_example.png?raw=true)


Notice that, since they are Imbalanced Datasets, there are some strategies used to overcome the problem of Imbalanced Classification.
For privacy reasons, the dataset cannot be published, but you can request at caterina.leonelli2@studio.unibo.it

For the model, 3 different neural networks were tested and analyzed:
• **U-net** 
• **VGG16**
• **VGG19**


# In Details
```
├──  main.py  - here's where program starts.
├──  params.py  - here's where argparse parses the user input in command line.
├──  train.py  - here's where model is trained or loaded with weights.
├──  predict.py  - here's were trained model is tested making predictions and evaluating them with metrics.
│
│
├──  models  
│    └── unet.py  - here's where basic unet model is defined.
│    └── vg16.py  - here's where vgg16 model is defined.
│    └── vgg19.py  - here's where vgg19 model is defined.
|
|
├──  utils  
│    └── fine_tuning.py  - here's where there are fine tuning functions for training.
│    └── graph.py  - here's where graph of the metrics are plotted.
│    └── load_dataset.py  - here's the datasets file that is responsible for all data loading.
│    └── print_details.py  - here's where print file report of the training and testing process.
│    └── load_dataset.py  - here's the datasets file that is responsible for all data loading.
│    └── metrics  
│        └── dice_coef.py  - dice coefficient.
│        └── iou.py  - intersection over union metric.
│        └── precision.py  - precision metric
│        └── recall.py  - recall (or sensitivity) metric.
│
│
├──  report  
│    └── final-report-english.pdf  - technical report of this project (English version).
     └── final-report-italian.pdf  - technical report of this project (Italian version). This is the official thesis.
```


# Contributing
Any kind of enhancement or contribution is welcomed. 
You can reach me out at caterina.leonelli2@studio.unibo.it
Please be kind: kindness is the best tool you can use to teach and learn.
