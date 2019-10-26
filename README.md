# keras-regression-cnns_House_Prices


Special thanks to [Adrian Rosebrock](https://www.pyimagesearch.com/author/adrian/)   for his  [great post](https://www.pyimagesearch.com/2019/01/28/keras-regression-and-cnns//) thaw was used as baseline for this tutourial.

This  simple code  creates and train a neural network to predict house prices based , you can try it by running the command  'python  housePrice_regression.py'. You can also run the code in your browser with the command 'ipython notebook housePrice_regression.ipynb'


The model used is as the following:


The dataset is from   https://github.com/emanhamed/Houses-dataset, this house dataset includes four numerical and categorical attributes as input and the one continous variable as output:
1. Number of bedrooms (continous)
2. Number of bathrooms(continous)
3. Area (continous)
4. Zip code (Cateogiral)
5.Price (continous)

Moreover the dataset includes 4 images for each house and this what will be used for training, The 4 images of each house will be tiled together into one image which will be the input to our CNN and the output is the price


When training finishes the following, curves will show the traning and validation. Another curve will also be shown for actual vs predicted prices. Both curves are saved to local drive. Also the trained  model is saved as housePrice.keras2 

