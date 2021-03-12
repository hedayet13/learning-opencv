from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import cv2 as cv



new_model = load_model('catAndDog50epochs.h5')


# predict on new images

dog_file ='CATS_DOGS/CATS_DOGS/test/DOG/9656.jpg'

img = cv.imread(dog_file)

from tensorflow.keras.preprocessing import image
dog_img = image.load_img(dog_file,target_size=(150,150))
dog_img = image.img_to_array(dog_img)

import numpy as np
dog_img = np.expand_dims(dog_img,axis = 0)

dog_img = dog_img/255
print(new_model.predict(dog_img))
res =new_model.predict_classes(dog_img)
print(res[0][0])
if res[0][0]==1:
    print("Its a dog ")
else:
    print("Its a pussy")
cv.imshow('image',img)

cv.waitKey(0)