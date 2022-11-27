import test
from keras.models import load_model

model = load_model("./oral_cancer_best_model.hdf5")

# Checking the Accuracy of the Model 
accuracy = model.evaluate_generator(generator= test.test_data)[1] 
print(f"The accuracy of the model is = {accuracy*100} %")