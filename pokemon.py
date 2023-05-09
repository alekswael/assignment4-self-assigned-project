##### IMPORTING DEPENDENCIES #####
# system tools and parse
import os 
import argparse
import warnings
warnings.filterwarnings("ignore")
# data tools
import pandas as pd
# tf tools
import tensorflow as tf
# image processsing
from tensorflow.keras.preprocessing.image import (ImageDataGenerator)
# VGG16 model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.vgg16 import VGG16
# layers
from tensorflow.keras.layers import (Dense, Flatten, Dropout, GlobalAveragePooling2D, BatchNormalization)
# generic model object
from tensorflow.keras.models import Model
# optimizers
from tensorflow.keras.optimizers import Adam, SGD
#scikit-learn
from sklearn.metrics import classification_report
import sklearn.model_selection as sk
# for plotting
import numpy as np
import matplotlib.pyplot as plt