import tensorflow as tf
from tensorflow import keras
import PIL
from PIL import Image
import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os import listdir
from matplotlib import image
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import SGD, Adagrad, RMSprop, Adam, Nadam
from keras.layers import Dropout
from keras.layers import Dense, Flatten, Activation, BatchNormalization, Convolution2D
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, Activation, MaxPooling2D, Dropout
from keras.optimizers import SGD, Adagrad, RMSprop, Adam, Nadam
from keras.models import model_from_json


foo = False
boo = True
if boo and not foo:
    print ('got your ass')