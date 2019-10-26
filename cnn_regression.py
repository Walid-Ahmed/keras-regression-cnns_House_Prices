# USAGE
# python cnn_regression.py 

# import the necessary packages
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
#from pyimagesearch import datasets
import numpy as np
import argparse
import locale
import os
from util import paths
import shutil 
import cv2
import numpy as np
from pathlib import Path
import itertools

from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model

#from pyimagesearch import datasets
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
import matplotlib.pyplot as plt
from keras.utils import plot_model




EPOCHS_NUM=200
# construct the path to the input .txt file that contains information
# on each house in the dataset and then load the dataset

print("[INFO] loading house attributes...")
inputPath =  "HousesInfo.txt"
cols = ["bedrooms", "bathrooms", "area", "zipcode", "price"]
df = pd.read_csv(inputPath, sep=" ", header=None, names=cols)
print(df.head())


#remove zipcounts that have kess than 25 houses
#Pandas Index.value_counts() function returns object containing counts of unique values. The resulting object will be in descending order so that the first element is the most frequently-occurring element. Excludes NA values by default.
zipcodeSeries=df["zipcode"].value_counts()  #<class 'pandas.core.series.Series'>
zipcodes = zipcodeSeries.keys().tolist()   #zipcodes as list
counts = zipcodeSeries.tolist()    #count of zipcodes as list  
for (zipcode, count) in zip(zipcodes, counts):
		# the zip code counts for our housing dataset is *extremely*
		# unbalanced (some only having 1 or 2 houses per zip code)
		# so let's sanitize our data by removing any houses with less
		# than 25 houses per zip code
		if count < 25:
			booleanVal=(df["zipcode"] == zipcode)  # this will be true at all zipcodes that should be deleted
			#print(type(booleanVal))   #<class 'pandas.core.series.Series'>
			idxs = df[booleanVal].index  #this will return indices of these true values
			df.drop(idxs, inplace=True)
print("[INFO]removed zipcodes which less than 25 houses")            






# load the house images and then 

if not os.path.exists('HouseImages'):
    os.makedirs('HouseImages')


#dividing images to foler accourding to house (4 imgaes/folder)
imagesPaths=paths.list_images("HousesDataset")
for imagePath in imagesPaths:
	sourcePath=imagePath
	imagePath=os.path.basename(imagePath)
	imageIndex=(imagePath.split("_"))[0]
	pathToSaveIamge=os.path.join("HouseImages",imageIndex)
	if not os.path.exists(pathToSaveIamge):
		os.makedirs(pathToSaveIamge)
	pathToSaveIamge=os.path.join("HouseImages",imageIndex,imagePath)
	shutil.copyfile(sourcePath, pathToSaveIamge)	


#read image and concotnat each 4 images into one image
trainingImages=[]
for recordIndex in df.index:
	dirOfImages=os.path.join("HouseImages",str(recordIndex))
	houseImages=[]
	files=os.listdir(dirOfImages)
	for file in sorted(files):
		if (file==".DS_Store"):
   			print(".DS_Store ignored" )
   			continue
		imgfilePath=os.path.join(dirOfImages,file)
		img=cv2.imread(imgfilePath)
		img = cv2.resize(img, (32, 32))
		houseImages.append(img)
	
	outputImage = np.zeros((64, 64, 3), dtype="uint8")
	outputImage[0:32, 0:32] = houseImages[0]
	outputImage[0:32, 32:64] = houseImages[1]
	outputImage[32:64, 32:64] = houseImages[2]
	outputImage[32:64, 0:32] = houseImages[3]
	trainingImages.append(outputImage)
	print("[INFO] Reading images from directory {}".format(dirOfImages))





#scale the pixel intensities to the range [0, 1]
images = np.array(trainingImages, dtype="float") / 255.0




# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainY, testY, trainX, testX) = train_test_split(df, images, test_size=0.25, random_state=42)

# find the largest house price in the training set and use it to
# scale our house prices to the range [0, 1] (will lead to better
# training and convergence)
maxPrice = trainY["price"].max()
print("maxPrice={}".format(maxPrice))
input("press any key")

trainY=trainY["price"].values
trainY = trainY / maxPrice

testY=testY["price"].values
testY = testY / maxPrice




print("Shapee of training  data set {}".format(trainX.shape))
print("Shapee of price  vector {}".format(testX.shape))

# create our Convolutional Neural Network and then compile the model
# using mean absolute percentage error as our loss, implying that we
# seek to minimize the absolute percentage difference between our
# price *predictions* and the *actual prices


inputShape = (64, 64, 3)
chanDim = -1
# define the model input
inputs = Input(shape=inputShape)
# CONV => RELU => BN => POOL
x = Conv2D(16, (3, 3), padding="same")(inputs)
x = Activation("relu")(x)
x = BatchNormalization(axis=chanDim)(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
# CONV => RELU => BN => POOL
x = Conv2D(32, (3, 3), padding="same")(x)
x = Activation("relu")(x)
x = BatchNormalization(axis=chanDim)(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
# CONV => RELU => BN => POOL
x = Conv2D(64, (3, 3), padding="same")(x)
x = Activation("relu")(x)
x = BatchNormalization(axis=chanDim)(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

# flatten the volume, then FC => RELU => BN => DROPOUT
x = Flatten()(x)
x = Dense(16)(x)
x = Activation("relu")(x)
x = BatchNormalization(axis=chanDim)(x)
x = Dropout(0.5)(x)

# apply another FC layer, this one to match the number of nodes
# coming out of the MLP
x = Dense(4)(x)
x = Activation("relu")(x)
x = Dense(1, activation="linear")(x)

# construct the CNN
model = Model(inputs, x)




#model = models.create_cnn(64, 64, 3, regress=True)
model.summary()
fileToSaveModelPlot='model.png'
plot_model(model, to_file='model.png')
print("[INFO] Model plot saved to {}".format(fileToSaveModelPlot) )







opt = Adam(lr=1e-3, decay=1e-3 / 200)
model.compile(loss="mean_absolute_percentage_error", optimizer=opt)

# train the model
print("[INFO] training model...")
history=model.fit(trainX, trainY, validation_data=(testX, testY),epochs=EPOCHS_NUM, batch_size=8)

# make predictions on the testing data





model.save("housePrice.keras2")
print("[INFO] model saved to housePrice.keras2")

# make predictions on the testing data
print("[INFO] predicting house prices...")
preds = model.predict(testX)



# compute the difference between the *predicted* house prices and the
# *actual* house prices, then compute the percentage difference and
# the absolute percentage difference
diff = preds.flatten() - testY
percentDiff = (diff / testY) * 100
absPercentDiff = np.abs(percentDiff)

# compute the mean and standard deviation of the absolute percentage
# difference
mean = np.mean(absPercentDiff)
std = np.std(absPercentDiff)

# finally, show some statistics on our model
locale.setlocale(locale.LC_ALL, "en_US.UTF-8")
print("[INFO] avg. house price: {}, std house price: {}".format(
	locale.currency(df["price"].mean(), grouping=True),
	locale.currency(df["price"].std(), grouping=True)))
print("[INFO] mean: {:.2f}%, std: {:.2f}%".format(mean, std))


#readjust house prices
testY=testY*maxPrice
preds=preds*maxPrice


validationLoss=(history.history['val_loss'])
trainingLoss=history.history['loss']




#------------------------------------------------
# Plot training and validation accuracy per epoch
epochs   = range(len(validationLoss)) # Get number of epochs
 #------------------------------------------------
plt.plot  ( epochs,     trainingLoss ,label="Training Loss")
plt.plot  ( epochs, validationLoss, label="Validation Loss" )
plt.title ('Training and validation loss')
plt.xlabel("Epoch #")
plt.ylabel("Loss")
fileToSaveAccuracyCurve="plot_acc.png"
plt.savefig("plot_acc.png")
print("[INFO] Loss curve saved to {}".format("plot_acc.png"))
plt.legend(loc="upper right")
plt.show()





#plot curves (Actual vs Predicted)
plt.plot  ( testY ,label="Actual price")
plt.plot  ( preds, label="Predicted price" )
plt.title ('House prices')
plt.xlabel("Point #")
plt.ylabel("Price")
plt.legend(loc="upper right")
plt.savefig("HousePrices.png")
plt.show()
print("[INFO] predicted vs actual price saved to HousePrices.png")








