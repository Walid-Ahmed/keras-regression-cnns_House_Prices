

import os

image_types = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff",".ppm")


def list_images(basePath, contains=None):
    # return the set of files that are valid
    return list_files(basePath, validExts=image_types, contains=contains)


def list_files(basePath, validExts=None, contains=None):
    # loop over the directory structure
    for (rootDir, dirNames, filenames) in os.walk(basePath):
        # loop over the filenames in the current directory
        for filename in filenames:
            # if the contains string is not none and the filename does not contain
            # the supplied string, then ignore the file
            if contains is not None and filename.find(contains) == -1:
                continue

            # determine the file extension of the current file
            ext = filename[filename.rfind("."):].lower()

            # check to see if the file is an image and should be processed
            if validExts is None or ext.endswith(validExts):
                # construct the path to the image and yield it
                imagePath = os.path.join(rootDir, filename)
                yield imagePath

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]           
            

def getTrainStatistics(datasetDir,train_dir,validation_dir):

    labels=get_immediate_subdirectories(train_dir)
    labels.sort()




    # Directory with our training cat/dog pictures
    train_label1_dir = os.path.join(train_dir, labels[0])
    train_label2_dir = os.path.join(train_dir, labels[1])

    # Directory with our validation cat/dog pictures
    validation_label1_dir = os.path.join(validation_dir, labels[0])
    validation_label2_dir = os.path.join(validation_dir, labels[1])

    """Now, let's see what the filenames look like in the `cats` and `dogs` `train` directories (file naming conventions are the same in the `validation` directory):"""

    train_label1_fnames = os.listdir( train_label1_dir )
    train_label2_fnames = os.listdir( train_label2_dir )

    #print(train_label1_fnames[:10])
    #print(train_label2_fnames[:10])

    """Let's find out the total number of cat and dog images in the `train` and `validation` directories:"""
    totalImages = len(os.listdir(train_label1_dir ) )+ len(os.listdir(train_label2_dir ) )+len(os.listdir( validation_label1_dir ) )+len(os.listdir( validation_label2_dir ) )
    print('[INFO] Total images in dataset '+datasetDir+ 'images :', totalImages)

    print('[INFO] Total training '+labels[0]+ ' images :', len(os.listdir(train_label1_dir ) ))
    print('[INFO] Total training ' + labels[1]+ ' images :', len(os.listdir(train_label2_dir ) ))
    NUM_TRAIN_IMAGES= len(os.listdir(train_label1_dir ))+len(os.listdir(train_label2_dir ) )

    print('[INFO] Total validation '+labels[0]+ ' images :', len(os.listdir( validation_label1_dir ) ))
    print('[INFO] Total validation '+ labels[1]+ ' images :', len(os.listdir( validation_label2_dir ) ))
    NUM_TEST_IMAGES=len(os.listdir( validation_label1_dir ) )+len(os.listdir( validation_label2_dir ) )

    print('[INFO] Total  training images in dataset: {} '.format(NUM_TRAIN_IMAGES))
    print('[INFO] Total validation images in dataset  {}'.format( NUM_TEST_IMAGES))                 

    return NUM_TRAIN_IMAGES,NUM_TEST_IMAGES

