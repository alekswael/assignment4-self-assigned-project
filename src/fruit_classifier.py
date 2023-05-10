##### IMPORTING DEPENDENCIES #####
# system tools and parse
import os 
import argparse
import warnings
warnings.filterwarnings("ignore")
# tf tools
import tensorflow as tf
# image processsing
from tensorflow.keras.preprocessing.image import (ImageDataGenerator)
# VGG16 model
from tensorflow.keras.applications.vgg16 import VGG16
# layers
from tensorflow.keras.layers import (Dense, Flatten, BatchNormalization)
# generic model object
from tensorflow.keras.models import Model
# optimizers
from tensorflow.keras.optimizers import SGD
#scikit-learn
from sklearn.metrics import classification_report
# for plotting
import numpy as np
import matplotlib.pyplot as plt

def input_parser(): # This is the function that parses the input arguments when run from the terminal.
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter) # This is the argument parser. I add the arguments below.
    ap.add_argument("-bs",
                    "--batch_size",
                    help="Batch size for training.",
                    type = int, default=32) # This is the argument for the batch size.
    ap.add_argument("-e",
                    "--epochs",
                    help="Number of epochs to train for.",
                    type = int, default=100) # This is the argument for the number of epochs.
    args = ap.parse_args() # Parse the args
    return args

img_size = (100, 100)

def setup_generators():
    # Parameters for loading data and images

    train_generator = ImageDataGenerator(horizontal_flip=True,
                                         rescale = 1./255,
                                         validation_split=0.2
                                         )
    
    test_generator = ImageDataGenerator(rescale=1./255)
    
    return train_generator, test_generator

def setup_data(train_generator, test_generator, batch_size_arg):
    # Split the data into three categories.
    train_ds = train_generator.flow_from_directory(
        directory=os.path.join(os.getcwd(), "fruits_v2", "train"),
        target_size=img_size,
        color_mode='rgb',
        class_mode='categorical',
        batch_size=batch_size_arg,
        shuffle=True,
        seed=42
    )
    
    val_ds = train_generator.flow_from_directory(
        directory=os.path.join(os.getcwd(), "fruits_v2", "val"),
        target_size=img_size,
        color_mode='rgb',
        class_mode='categorical',
        batch_size=batch_size_arg,
        shuffle=True,
        seed=42
    )

    test_ds = test_generator.flow_from_directory(
        directory=os.path.join(os.getcwd(), "fruits_v2", "test"),
        target_size=img_size,
        color_mode='rgb',
        class_mode='categorical',
        batch_size=batch_size_arg,
        shuffle=False,
        seed=42
    )
    
    return train_ds, val_ds, test_ds

def model_setup():

    tf.keras.backend.clear_session()
    
    # load model without classifier layers
    model = VGG16(include_top=False, 
                pooling="avg",
                input_shape=(img_size[0], img_size[1], 3),
                weights='imagenet')

    # mark loaded layers as not trainable
    for layer in model.layers:
        layer.trainable = False
    
    x = Flatten()(model.layers[-1].output)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(33, activation='softmax')(x)

    model = Model(inputs=model.inputs, outputs=outputs)

    # compile
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.01,
        decay_steps=10000,
        decay_rate=0.9)
    
    sgd = SGD(learning_rate=lr_schedule)

    model.compile(optimizer=sgd,
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    
    return model

def train_model(model, train_ds, val_ds, epochs_arg):
    
    # Define the EarlyStopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                                    patience=2,
                                                    restore_best_weights=True)
    
    history = model.fit_generator(train_ds,
                        validation_data = val_ds,
                        epochs=epochs_arg,
                        callbacks=[early_stopping]
                        )
    
    # Getting the amount of epochs trained
    actual_epochs = len(history.history['loss'])
    
    return history, actual_epochs

##### PLOTTING FUNCTION #####
def plot_history(H, epochs):
    plt.style.use("seaborn-colorblind")

    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss", linestyle=":")
    plt.title("Loss curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc", linestyle=":")
    plt.title("Accuracy curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.legend()
    plt.savefig(os.path.join(os.getcwd(), "out", "classification.png"))
    
def make_predictions(model, test_ds):
    y_test = test_ds.classes

    y_predictions = model.predict_generator(test_ds, steps=len(test_ds))

    y_pred = np.argmax(y_predictions, axis=1)
    
    return y_test, y_predictions, y_pred

def print_report(y_test, y_pred, test_ds):
    # Get the classification report
    report = classification_report(y_test,
                                   y_pred,
                                   target_names = test_ds.class_indices.keys()
                                   )
    # Save the report
    with open(os.path.join(os.getcwd(), "out", "classification_report.txt"), "w") as f:
            f.write(report)
    # Print the report
    print(report)
    
def main():
    args = input_parser() # Parse the input arguments.
    print("Loading and preprocessing data...")
    print("Setting up generators and data...")
    train_generator, test_generator = setup_generators()
    print("Setting up data...")
    train_ds, val_ds, test_ds = setup_data(train_generator, test_generator, args.batch_size)
    print("Setting up model...")
    model = model_setup()
    print("Training model...")
    history, actual_epochs = train_model(model, train_ds, val_ds, args.epochs)
    print("Plotting and saving learning curves...")
    plot_history(history, actual_epochs)
    print("Making predictions...")
    y_test, y_predictions, y_pred = make_predictions(model, test_ds)
    print("Printing classification report...")
    print_report(y_test, y_pred, test_ds)

if __name__ == "__main__":
    main()