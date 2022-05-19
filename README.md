# Assignment 4 – Human Action classification
The portfolio for __Visual Analytics S22__ consists of 4 projects (3 class assignments and 1 self-assigned project). This is the __fourth and final assignment__ in the portfolio.


## 1. Contribution
The code was written independently. 

The code provided when downloading the data (as well as [here](https://www.kaggle.com/code/emirhanai/human-action-detection-artificial-intelligence-cnn)) was used to help me load and process the data. I was also inspired by their use of a [EarlyStopping](https://github.com/agnesbn/VIS_assignment4/blob/90b1211085e8b418830b2a70cbbc9618b76ce2d3/src/human_action_classification.py#L164) function, so that the model would stop training if there is no more to improve on.

I also found [this simple example](https://gist.github.com/RyanAkilos/3808c17f79e77c4117de35aa68447045) of how to get and process data, train a model and finally save a classification report when the data has been loaded using `flow_from_directory()` extremely useful. During the course, we only saw how to do these tasks when `X` and `y` had been loaded seperately.

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

The data was already split between training (15000 images) and testing (3000 images) data.

The goal for the project was then to see how well I could train a model to do classification of the images using the provided labels. I decided to use the pre-trained model, VGG16, and only do training for the classifier layers. This was the 

The provided code, [human_action_classification.py](https://github.com/agnesbn/VIS_assignment4/blob/main/src/human_action_classification.py), loads and processes the data. It then initialises the VGG16-model without its classifier layers and adds new trainable classifier layers. Finally, the model is trained on the data and the results are saved in form of a classification report and a plot of the history.

## 3. Usage
### Install packages
Before running the script, run the following from the command line:
```
pip install --upgrade pip
pip install opencv-python scikit-learn tensorflow tensorboard tensorflow_hub pydot scikeras[tensorflow-cpu] tensorflow_datasets
sudo apt-get update
sudo apt-get -y install graphviz
```

### Get the data
- Download the data from: https://www.kaggle.com/datasets/7ee7818707be9f1b0dfc4703c1fba3fdab24174b5acca2d3e4e2c3de82152ae6.
- Unzip it and place it in the [`in`](https://github.com/agnesbn/VIS_assignment4/tree/main/in) folder.
- Make sure the path to the data is: `VIS_assignment4/in/archive/emirhan_human_dataset/datasets/human_data` or change the path in line [111](https://github.com/agnesbn/VIS_assignment4/blob/90b1211085e8b418830b2a70cbbc9618b76ce2d3/src/human_action_classification.py#L111) and [112](https://github.com/agnesbn/VIS_assignment4/blob/90b1211085e8b418830b2a70cbbc9618b76ce2d3/src/human_action_classification.py#L112) to reflect the structure of your repository.

### Run the script
Make sure your current directory is the `VIS_assignment4` folder. Then run:
```
python src/human_action_classification.py (--epochs <EPOCHS> --plot_name <PLOT NAME> --report_name <REPORT NAME>
```

* `<EPOCHS>` represents the number of epochs that the model trains in. The default value is `50`.
* `<PLOT NAME>` represents the name that the history plot is saved under (it will always be saved as a PNG). The default value is `history_plot`.
* `<REPORT NAME>` represents the name that the classification report is saved under (it will always be saved as a TXT). The default value is `classification_report`.

The results are saved in the [`out`](https://github.com/agnesbn/VIS_assignment4/tree/main/out) folder.

## 4. Discussion of results
After training the model for 50 epochs, an **accuracy of 47%** was reached. Though the training loss and accuracy curves do seem to have flattened a bit, they may still improve with longer training time. The validation loss and accuracy curves, however, seem to have flattened quite a lot more around the 10-15 epochs mark.

![](https://github.com/agnesbn/VIS_assignment4/blob/main/out/history_plot.png)

The person who uploaded the data trained a number of CNNs on the data and only reached a **test accuracy of 43.17%** when training on VGG16 for 50 epochs. This indicates that getting accuracy scores much higher than mine is unlikely.
