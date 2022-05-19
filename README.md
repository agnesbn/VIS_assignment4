# Assignment 4 – Human Action classification
The portfolio for __Visual Analytics S22__ consists of 4 projects (3 class assignments and 1 self-assigned project). This is the __fourth and final assignment__ in the portfolio.


## 1. Contribution
The code was written independently. 

The code provided by when downloading the data (as well as [here](https://www.kaggle.com/code/emirhanai/human-action-detection-artificial-intelligence-cnn)) was used to help me load and process the data.

I also found [this simple example](https://gist.github.com/RyanAkilos/3808c17f79e77c4117de35aa68447045) of how to get and process data, train a model and finally save a classification report when the data has been loaded using `flow_from_directory()` and not by loading `X` and `y` seperately as we saw during the course.

## 2. Methods
For this self-assigned project, I wanted to use the methods introduced to us during the Visual Analytics course to train a model to do classification of different human actions. The data, which was found on [Kaggle](https://www.kaggle.com/datasets/7ee7818707be9f1b0dfc4703c1fba3fdab24174b5acca2d3e4e2c3de82152ae6), provided images of 15 different human actions:
* calling
* clapping
* cycling
* dancing
* drinking
* eating
* fighting
* hugging
* laughing
* listening_to_music
* running
* sitting
* sleeping
* texting
* using_laptop

The data is already split between the training data (with 15000 images) and the testing data (with 3000 images).

The goal for the project was then to see how well I could train a model to do classification of the images using the provided labels. I decided to use the pre-trained model, VGG16, and only do training for the classifier layers. This was the 

## 3. Usage
### Get the data
- Download the data from: https://www.kaggle.com/datasets/7ee7818707be9f1b0dfc4703c1fba3fdab24174b5acca2d3e4e2c3de82152ae6.
- Unzip it and place it in the "in" folder.
- Make sure the path to the data is: "VIS_assignment4/in/human-action-detection-artificial-intelligence/emirhan_human_dataset/datasets/human_data"
    - Otherwise, change the code in line **XXX** to fit with your path.
    - 
### Install packages
Before running the script, run the following in the Terminal:
```
pip install --upgrade pip
pip install opencv-python scikit-learn tensorflow tensorboard tensorflow_hub pydot scikeras[tensorflow-cpu] tensorflow_datasets
sudo apt-get update
sudo apt-get -y install graphviz
```

### Run the script
Then, from the `VIS_assignment1` directory, run:


## 4. Discussion of results

