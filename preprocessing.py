import cv2
import os
import shutil 
import math
import random
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings("ignore")



# Function to remove Duplicate Images in the Dataset
def findDelDuplImg(file_name , file_dir):
    searchedImgPath = os.path.join(file_dir, file_name);
    searchedImage = np.array(cv2.imread(searchedImgPath, 0));
    # Start iterate over all images
    for cmpImageName in os.listdir(file_dir):
        if cmpImageName != file_name:
            # If name is different
            try:
                # Concatenate path to image
                cmpImagePath = os.path.join(file_dir, cmpImageName);
                # Open image to be compared
                cmpImage = np.array(cv2.imread(cmpImagePath, 0))
                # Count root mean square between both images (RMS)
                rms = math.sqrt(mean_squared_error(searchedImage, cmpImage))
            except:
                continue
            # If RMS is smaller than 3 - this means that images are similar or the same
            if rms < 3:
                # Delete Same Image in Dir
                os.remove(cmpImagePath);
                
# Function for Image preprocessing
def processDataset(dataset_src, dataset_dest):
    # Making a Copy of Dataset
    shutil.copytree(src, dest)
    for folder in os.listdir(dest):
        for (index, file) in enumerate(os.listdir(os.path.join(dest, folder)), start = 1):
            filename = f'img_{folder}_{index}.jpg';
            img_src = os.path.join(dest, folder, file);
            img_des = os.path.join(dest, folder, filename);
            # Preprocess the Images
            img = cv2.imread(img_src);
            img = cv2.resize(img, (256, 256));
            img = cv2.copyMakeBorder(img, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0);
            img = cv2.blur(img, (2, 2));
            cv2.imwrite(img_des ,img);
            os.remove(img_src);
        for file in os.listdir(os.path.join(dest, folder)):
                # Find duplicated images and delete duplicates.
                findDelDuplImg(file, os.path.join(dest, folder));

# Source Location for Dataset
src = 'Input\OralCancer';
# Destination Location for Dataset
dest = 'OralCancer';
# Image preprocessing
processDataset(src, dest);