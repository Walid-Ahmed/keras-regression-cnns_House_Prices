# USAGE
# python cnn_regression.py --dataset Houses-dataset/Houses\ Dataset/

# import the necessary packages
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from pyimagesearch import datasets
from pyimagesearch import models
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



# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, required=True,
	help="path to input dataset of house images")
args = vars(ap.parse_args())

# construct the path to the input .txt file that contains information
# on each house in the dataset and then load the dataset
print("[INFO] loading house attributes...")
inputPath = os.path.sep.join([args["dataset"], "HousesInfo.txt"])
df= datasets.load_house_attributes(inputPath)


recordsIndicies=df.index.values




# load the house images and then scale the pixel intensities to the
# range [0, 1]

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
for recordIndex in recordsIndicies:
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






images = np.array(trainingImages, dtype="float") / 255.0

print("Shapee of images data set {}".format(images.shape))
print("Shapee of price  vector {}".format(df.shape))






'''
print("[INFO] loading house images...")
images = datasets.load_house_images(df, args["dataset"])
images = images / 255.0
'''

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
split = train_test_split(df, images, test_size=0.25, random_state=42)
(trainAttrX, testAttrX, trainImagesX, testImagesX) = split

# find the largest house price in the training set and use it to
# scale our house prices to the range [0, 1] (will lead to better
# training and convergence)
maxPrice = trainAttrX["price"].max()
trainY = trainAttrX["price"] / maxPrice
testY = testAttrX["price"] / maxPrice

# create our Convolutional Neural Network and then compile the model
# using mean absolute percentage error as our loss, implying that we
# seek to minimize the absolute percentage difference between our
# price *predictions* and the *actual prices*
model = models.create_cnn(64, 64, 3, regress=True)
opt = Adam(lr=1e-3, decay=1e-3 / 200)
model.compile(loss="mean_absolute_percentage_error", optimizer=opt)

# train the model
print("[INFO] training model...")
model.fit(trainImagesX, trainY, validation_data=(testImagesX, testY),
	epochs=200, batch_size=8)

# make predictions on the testing data
print("[INFO] predicting house prices...")
preds = model.predict(testImagesX)

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