# FashionTag

This is a consulting project to develop a web application to automatically creating fashion tags for images uploaded. The training data is 1 million labelled fashion images from a CVPR 2018 workshop. I used transfer learning to create embedding data of the fashion images to achieve more than 100 times accerlation in the deep neural network training process. Flask was used to develop the web application. The final product is dockerized and can be pull from docker hub. 


## Getting Started

To get the data, please go to the 'Data' folder in this repo for further instructions. To train your own model, you will first need to generate embedding data for the images and store the embedding in your data folder. This will acceccrate your deep neural network training process by trading time complexcity with space complexicity. 

All scripts should be ran from the project root directory, e.g.:

```
python scripts/train.py
```
### Prerequisites

I used tensorflow and keras for the model development on AWS enviroment tensorflow_p36. To install a similar enviroment, you can run pip to install the following packages. I also installed keras-metrics, a package to use customerized metrics to monitor the training process. For the Flask development, you can use the docker file to setup the enviroment.

```
pip install tensorflow==1.13.1
pip install Keras==2.2.4
pip install Keras-Applications==1.0.7
pip install keras-metrics==1.1.0
pip install Keras-Preprocessing==1.0.9
```

## Running the code

The code should be execute in the correct sequence. 

### Download and clean the data

To download the data, you should first download the [training.json file](https://www.kaggle.com/c/imaterialist-challenge-fashion-2018), then run the following code to (1) remove broken links from the json files; (2) download the images using the url links from the json file; (3) create label data

```
python delete_broken_images.py
python scripts/download_images.py ./data/train.json ./data/train
python scripts/create_label.py
```

### Create image embedding data using the first half of the model

The pretrained CNN model is very large, making the total training process slow eventhough we freeze the weights of the pre-trained model for transfer learning. As the model weight is not being trained, I decided to extract the output of the pretrained CNN model (VGG19) and store it temportatly and use it as embedding data for input for the fully connective network. This way, it is more than hundreads time faster in training. 
```
python scripts/create_embedding.py
```
### Train the second half of the model using multiple customerized loss function

Traditionally, a multilabel classification problem will use binary cross-entropy as the cost function. However, due to the sparcity of our target data, in other words, most columns in y are 0. The model can just output all zero to get a high accuracy. This makes it very difficult to train. To solve this problem, instead of focusing on the accurcay, I focus on pushing the F1 score of the model, which is a complimize between recall and precision. Therefore, I customerized a loss function based on the F1 score to train the model. Another difficulty of this project is that our data is highly imbalanced even after label selection. Focal loss [https://arxiv.org/abs/1708.02002], is a method developed in the object detection task. In object detection, often, there are a large background, which is easy to identify but occupy most of the data. The model saturated easily in very accuracily predicting a background. However, what we really interested is the a few obejct, which is our positive examples. Focal loss was designed to force the model focus on the few positive examples (outputting 1, not 0). This is similar to our imbalanced class and sparse target data problems. Therefore, I also customerized a focal loss function as our loss function in this project to increase the F1 score.

```
python scripts/train.py
```

## Deployment

I have Flask to deploy the deep learning model to a web application. For more information, please visit [my github page of Flask and Docker](https://github.com/YIZHE12/keras-flask-deploy-webapp). To download and run the docker, you can run the following commands:

```
docker pull echeng1212/keras_flask_app:insightdemo
docker run -d -p 5000:5000 echeng1212/keras_flask_app:insightdemo
```


## Built With

* [Flask](http://flask.pocoo.org/) - The web framework used


## Contact information

* **Yi Zheng** :echeng1212@gmail.com 


## Acknowledgments

* Inspiration: 
https://github.com/KDercksen/hunter2_fashion




