from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import cv2 as cv



new_model = load_model('catAndDog50epochs.h5')



# predict on new images

dog_file ='CATS_DOGS/CATS_DOGS/test/DOG/9650.jpg'
from tensorflow.keras.preprocessing import image
dog_img = image.load_img(dog_file,target_size=(150,150))
dog_img = image.img_to_array(dog_img)

import numpy as np
dog_img = np.expand_dims(dog_img,axis = 0)

dog_img = dog_img/255
print(new_model.predict(dog_img))
print(new_model.predict_classes(dog_img))