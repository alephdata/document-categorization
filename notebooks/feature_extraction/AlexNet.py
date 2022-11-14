import pandas as pd
import matplotlib as plt
import numpy as np
import os

## For computers with no GPU. Remove if GPU is present.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

## Folder names inside our data directory
dataset_path = os.listdir("Tobacco3482-jpg")
print(dataset_path)

## Different document classes in our data directory.
doc_types = os.listdir("Tobacco3482-jpg")
print(doc_types)

print("Types of documents found:", len(dataset_path))

#########################################################################################
#########################################################################################

docs = []
# Get all image file names + their type.
for item in doc_types:
    all_docs = os.listdir("Tobacco3482-jpg/" + item)

    # Add them to the list:
    for doc in all_docs:
        docs.append((item, str("Tobacco3482-jpg/" + item) + "/" + doc))

# print(docs)
# docs

#########################################################################################
#########################################################################################

# Build a dataframe:
docs_df = pd.DataFrame(data=docs, columns=["doc type", "image"])
print(docs_df.head())
print("Total number of scanned pages in the dataset:", len(docs_df))

#########################################################################################
#########################################################################################

## We count how many images correspond to each
## document type.
doc_count = docs_df["doc type"].value_counts()

## We print the results.
print("Documents in each category:")
print(doc_count)

#########################################################################################
#########################################################################################

## Install opencv in environment.
import cv2

path = "Tobacco3482-jpg/"
im_size = 227

images = []
labels = []
type_counter = 0
file_counter = 0
faulty = []


## In this loop, we extract all document images,
## and resize them to 227x227x3 (color image)
for i in doc_types:
    data_path = path + str(i)
    filenames = [i for i in os.listdir(data_path)]
    for f in filenames:
        img = cv2.imread(data_path + "/" + f)  # reading image as array
        v_type = type(img)
        if v_type is np.ndarray:
            # print(f," : ", img.shape[0], " x ", img.shape[1])
            img = cv2.resize(img, (im_size, im_size))
            images.append(img)
            labels.append(i)

        ## Save the positions of damaged files:
        if v_type is not np.ndarray:
            type_counter = type_counter + 1
            faulty.append(file_counter)
            # print( "Faulty file!")
        file_counter = file_counter + 1


print("There are a total of ", type_counter, " faulty files.")

#########################################################################################
#########################################################################################

## We convert images into a darray:
images = np.array(images)
print("Size of image tensor is:", images.shape)

## We resize darray entries to be between 0 and 1.
images = images.astype("float32") / 255.0

#########################################################################################
#########################################################################################

## Eliminate damaged files from database.
print("The original size of our database is:", docs_df.shape)
print("We want to eliminate the rows: ", faulty)
docs_df = docs_df.drop(labels=faulty, axis=0)
# docs_df2 = docs_df.reindex(axis=0)
print("The new dimensions are:", docs_df.shape)

#########################################################################################
#########################################################################################

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

## To train the CNN, we need to encode our response as
## a 10-long vector with 0's and 1's. For this, we use
## labelencoder + onehotencoder.
y = docs_df["doc type"].values
# print(y[:5])

y_labelencoder = LabelEncoder()
y = y_labelencoder.fit_transform(y)
# print(y[:5])

y = y.reshape(-1, 1)
# print(y[:5])

onehotencoder = OneHotEncoder(categories="auto", sparse=False)
Y = onehotencoder.fit_transform(y)
print("The encoded responses have dimension:", Y.shape)
print("The first 5 rows of encoded responses:")
print(Y[:5, :])

#########################################################################################
#########################################################################################

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

## We shuffle the images:
images, Y = shuffle(images, Y, random_state=5)

## Generate training/testing datasets.
train_x, test_x, train_y, test_y = train_test_split(images, Y, test_size=0.1, random_state=1)

# Display dimensions of training/testing sets:
print("The dimensions of the training/testing sets.")
print("Training images:", train_x.shape)
print("Training classes:", train_y.shape)
print("Testing images:", test_x.shape)
print("Testing classes:", test_y.shape)

#########################################################################################
#########################################################################################

## Note: Original code uses tensorflow v1,
##      but local installation is v2. To
##      correct this, we load tensorflow.compat,
##      which is a module of TF that contains v1.
##      to use v1, we use tensorflow.compat.v1
from tensorflow import compat as tf

# placeholders are not executable immediately so we need to disable eager exicution in TF 2 not in 1
tf.v1.disable_eager_execution()

num_classes = 10
x = tf.v1.placeholder(tf.v1.float32, shape=[None, 227, 227, 3])
y_ = tf.v1.placeholder(tf.v1.float32, [None, num_classes])

#########################################################################################
#########################################################################################

## We create a dictionary with the weights/biases needed for AlexNet CNN.
weights = {
    "w1": tf.v1.Variable(tf.v1.random_normal(shape=[11, 11, 3, 96], dtype=tf.v1.float32), name="w1"),
    "w2": tf.v1.Variable(tf.v1.random_normal(shape=[5, 5, 96, 256], dtype=tf.v1.float32), name="w2"),
    "w3": tf.v1.Variable(tf.v1.random_normal(shape=[3, 3, 256, 384], dtype=tf.v1.float32), name="w3"),
    "w4": tf.v1.Variable(tf.v1.random_normal(shape=[3, 3, 384, 384], dtype=tf.v1.float32), name="w4"),
    "w5": tf.v1.Variable(tf.v1.random_normal(shape=[3, 3, 384, 256], dtype=tf.v1.float32), name="w5"),
    "wfc1": tf.v1.Variable(tf.v1.random_normal(shape=[6 * 6 * 256, 4096], dtype=tf.v1.float32), name="wfc1"),
    "wfc2": tf.v1.Variable(tf.v1.random_normal(shape=[4096, 4096], dtype=tf.v1.float32), name="wfc2"),
    "wout": tf.v1.Variable(tf.v1.random_normal(shape=[4096, num_classes], dtype=tf.v1.float32), name="wout"),
}
biases = {
    "b1": tf.v1.Variable(tf.v1.random_normal(shape=[96]), name="b1"),
    "b2": tf.v1.Variable(tf.v1.random_normal(shape=[256]), name="b2"),
    "b3": tf.v1.Variable(tf.v1.random_normal(shape=[384]), name="b3"),
    "b4": tf.v1.Variable(tf.v1.random_normal(shape=[384]), name="b4"),
    "b5": tf.v1.Variable(tf.v1.random_normal(shape=[256]), name="b5"),
    "bfc1": tf.v1.Variable(tf.v1.random_normal(shape=[4096]), name="bfc1"),
    "bfc2": tf.v1.Variable(tf.v1.random_normal(shape=[4096]), name="bfc2"),
    "bout": tf.v1.Variable(tf.v1.random_normal(shape=[num_classes]), name="bout"),
}

#########################################################################################
#########################################################################################

## We define the AlexNet CNN:
def alex_net(x, weights, biases):
    x = tf.v1.reshape(x, shape=[-1, 227, 227, 3])
    print("*******************************************")
    print("The size of x is: ", x.shape)

    ## 1st convolutional + maxpool layers
    conv1_in = tf.v1.nn.conv2d(x, weights["w1"], strides=[1, 4, 4, 1], padding="SAME")
    conv1_in = tf.v1.nn.bias_add(conv1_in, biases["b1"])
    conv1 = tf.v1.nn.relu(conv1_in)

    maxpool1 = tf.v1.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")
    print("Size after 1st convolutional layer: ", maxpool1.shape)

    ## 2nd convolutional + maxpool layers
    conv2_in = tf.v1.nn.conv2d(maxpool1, weights["w2"], strides=[1, 1, 1, 1], padding="SAME")
    conv2_in = tf.v1.nn.bias_add(conv2_in, biases["b2"])
    conv2 = tf.v1.nn.relu(conv2_in)

    maxpool2 = tf.v1.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")
    print("Size after 2nd convolutional layer: ", maxpool2.shape)

    ## 3rd convolutional layer
    conv3_in = tf.v1.nn.conv2d(maxpool2, weights["w3"], strides=[1, 1, 1, 1], padding="SAME")
    conv3_in = tf.v1.nn.bias_add(conv3_in, biases["b3"])
    conv3 = tf.v1.nn.relu(conv3_in)
    print("Size after 3rd convolutional layer: ", conv3.shape)

    ## 4th convolutional layer
    conv4_in = tf.v1.nn.conv2d(conv3, weights["w4"], strides=[1, 1, 1, 1], padding="SAME")
    conv4_in = tf.v1.nn.bias_add(conv4_in, biases["b4"])
    conv4 = tf.v1.nn.relu(conv4_in)
    print("Size after 4th convolutional layer: ", conv4.shape)

    ## 5th convolutional + maxpool layer:
    conv5_in = tf.v1.nn.conv2d(conv4, weights["w5"], strides=[1, 1, 1, 1], padding="SAME")
    conv5_in = tf.v1.nn.bias_add(conv5_in, biases["b5"])
    conv5 = tf.v1.nn.relu(conv5_in)

    maxpool5 = tf.v1.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")
    print("Size after 5th convolutional layer: ", maxpool5.shape)

    ## 6th flat layer.
    fc6 = tf.v1.reshape(maxpool5, [-1, weights["wfc1"].get_shape().as_list()[0]])
    fc6 = tf.v1.add(tf.v1.matmul(fc6, weights["wfc1"]), biases["bfc1"])
    fc6 = tf.v1.nn.relu(fc6)
    print("Size after reshaping the image is: ", fc6.shape)

    ## 7th flat layer:
    fc7 = tf.v1.nn.relu_layer(fc6, weights["wfc2"], biases["bfc2"])
    print("Size after reshaping the image is: ", fc7.shape)

    ## 8th output layer:
    fc8 = tf.v1.add(tf.v1.matmul(fc7, weights["wout"]), biases["bout"])
    out = tf.v1.nn.softmax(fc8)

    return out


#########################################################################################
#########################################################################################

## Create the model:
model = alex_net(x, weights, biases)
print(model)

#########################################################################################
#########################################################################################

## Establish learning rate, cost and optimizer algorithm.
learning_rate = 0.05
cost = tf.v1.reduce_mean(tf.v1.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=y_))
optimizer = tf.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

## Initialize all the weights and biases of the model.
init = tf.v1.global_variables_initializer()

#########################################################################################
#########################################################################################

import random

# Display dimensions of training/testing sets:
print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)

## Define number of epochs and start session
cost_history = []
n_epochs = 100
sess = tf.v1.Session()
sess.run(init)


## We run for 100 epochs:
for i in range(n_epochs):
    print("Starting Epoch ", i)

    ## Select a batch of size 20
    batchIndex = random.sample(range(3133), 20)
    train_x_ = train_x[batchIndex, :]
    train_y_ = train_y[batchIndex, :]

    ## Update weights/biases with this batch.
    a, c = sess.run([optimizer, cost], feed_dict={x: train_x_, y_: train_y_})
    cost_history = np.append(cost_history, c)
    print("Epoch ", i, " - ", "Cost: ", c)

#########################################################################################
#########################################################################################
