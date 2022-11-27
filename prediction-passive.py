from keras.preprocessing import image
from keras.models import load_model
import numpy as np

model = load_model("./oral_cancer_best_model.hdf5")

def cancerPrediction(path):
    # Loading Image
    img = image.load_img(path, target_size=(256,256))
    # Normalizing Image
    norm_img = image.img_to_array(img)/255
    # Converting Image to Numpy Array
    input_arr_img = np.array([norm_img])
    # Getting Predictions
    pred = (model.predict(input_arr_img) > 0.5).astype(int)[0][0]
    # Printing Model Prediction
    if pred == 0:
        print("Cancer")
    else:
        print("Non-Cancer")
    
# Path for the image to get predictions    
path = "./input/OralCancer/cancer/01960a64-cfe8-444d-bbc5-575c15389a21.jpg"
path1 = "./input/OralCancer/non-cancer/20200314_1130302.jpg"
cancerPrediction(path)
cancerPrediction(path1)