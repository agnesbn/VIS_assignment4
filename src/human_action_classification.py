"""
Human Action classification with VGG16
"""
""" Import the relevant packages """
# Import relevant packages
 # base tools
import os
import glob
 # argument parser
import argparse
 # data analysis
import numpy as np
import pandas as pd
 # plotting
import seaborn as sns
import matplotlib.pyplot as plt
import pathlib
 # from scikit learn
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
 # image processing
import PIL
import PIL.Image
 # tensorflow
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.layers import *
from tensorflow import keras
#from tensorflow.keras import Sequential
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model, Sequential
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import Callback, EarlyStopping
import tensorflow_hub as hub
from tensorflow.keras.optimizers import SGD, Adam
 # scikeras wrapper
from scikeras.wrappers import KerasClassifier
 # warnings
import warnings
warnings.filterwarnings("ignore")


""" Basic functions """
# Argument parser
def parse_args():
    ap = argparse.ArgumentParser()
    # number of epochs
    ap.add_argument("-e",
                    "--epochs",
                    default=50,
                    type=int,
                    help = "The number of epochs to train your model in")
    # report name argument
    ap.add_argument("-r",
                    "--report_name",
                    type=str,
                    default="classification_report",
                    help="The name of the classification report")
    # plot name argument
    ap.add_argument("-p",
                    "--plot_name",
                    type=str,
                    default="history_plot",
                    help="The name of the plot of loss and accuracy")
    args = vars(ap.parse_args())
    return args 

# Function to save history
def save_history(H, epochs, plot_name):
    outpath = os.path.join("out", f"{plot_name}.png")
    plt.style.use("seaborn-colorblind")
    
    plt.figure(figsize=(12,6))
    plt.suptitle(f"Human Action Classification", fontsize=16)
    plt.subplot(1,2,1)
    plt.plot(np.arange(0, epochs), H.history["loss"], label="Train")
    plt.plot(np.arange(0, epochs), H.history["val_loss"], label="Validation", linestyle=":")
    plt.title("Loss curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.legend()
    
    plt.subplot(1,2,2)
    plt.plot(np.arange(0, epochs), H.history["accuracy"], label="Train")
    plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="Validation", linestyle=":")
    plt.title("Accuracy curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.legend()
    plt.savefig(os.path.join(outpath))

    
# Function for saving classification report
def report_to_txt(report, report_name, epochs):
    outpath = os.path.join("out", f"{report_name}.txt")
    with open(outpath,"w") as file:
        file.write(f"Classification report\nData: Human Action Detection - Artificial Intelligence\nModel: VGG16\nEpochs: {epochs}\n")
        file.write(str(report))    
        

""" Human action classification """
def classification(epochs, report_name, plot_name):
    # define useful values
    train_data_path = os.path.join("in", "archive", "emirhan_human_dataset", "datasets", "human_data", "train_data")
    test_data_path = os.path.join("in", "archive", "emirhan_human_dataset", "datasets", "human_data", "test_data")
    img_rows = 128
    img_cols = 128
    epochs = epochs
    batch_size = 128
    num_of_train_samples = 15000
    num_of_test_samples = 3000
    
    # create image data generators for training and testing data
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       rotation_range=40,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True,
                                       fill_mode='nearest')
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    
    # get data from directories
    train_generator = train_datagen.flow_from_directory(train_data_path,
                                                        target_size=(img_rows, img_cols),
                                                        batch_size=batch_size,
                                                        class_mode='categorical',
                                                        shuffle=True)
    validation_generator = test_datagen.flow_from_directory(test_data_path,
                                                            target_size=(img_rows, img_cols),
                                                            batch_size=batch_size,
                                                            class_mode='categorical',
                                                            shuffle = False)
    # initialise the model
    pre_model = VGG16(input_shape=(img_rows,img_cols,3),
                      include_top=False,
                      weights='imagenet',
                      pooling='avg')
    pre_model.trainable = False
    # model inputs
    inputs = pre_model.input
    # add new classifier layers
    flat1 = Flatten()(pre_model.layers[-1].output)
    x = Dense(256, activation='relu')(flat1)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    outputs = Dense(15, activation='softmax')(x)
    # define new model
    model = Model(inputs=inputs, outputs=outputs)
    # compile model
    model.compile(loss = 'categorical_crossentropy',
                  optimizer='Adam',
                  metrics=['accuracy'])
    # stop training if validation loss is 0 for 5 epochs 
    my_callbacks  = [EarlyStopping(monitor='val_loss',
                                   min_delta=0,
                                   patience=5, 
                                   mode='auto')]
    # train model
    H = model.fit_generator(train_generator,
                            steps_per_epoch=num_of_train_samples // batch_size,
                            epochs=epochs,
                            validation_data=validation_generator,
                            validation_steps=num_of_test_samples // batch_size)
    
    validation_generator.reset()
    # evaluate model
    Y_pred = model.predict_generator(validation_generator, 
                                     num_of_test_samples // batch_size+1)
    y_pred = np.argmax(Y_pred, axis=1)
    print('Classification Report')
    # get labels
    target_names = list(train_generator.class_indices.keys())
    # create classification report
    report = classification_report(validation_generator.classes, 
                                   y_pred, 
                                   target_names=target_names)
    # get actual number of epochs used
    n_epochs = len(H.history['loss'])
    # save report
    report_to_txt(report, report_name, n_epochs)
    # save history plot
    save_history(H, n_epochs, plot_name)
    return print(report)
    

""" Main function """
def main():
    # parse arguments
    args = parse_args()
    # get arguments
    epochs = args["epochs"]
    report_name = args["report_name"]
    plot_name = args["plot_name"]
    # train model, save plot and classification report
    classification(epochs, report_name, plot_name)
    
    
if __name__=="__main__":
    main()
