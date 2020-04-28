import matplotlib.pyplot as plt
import cv2 as cv

cat4 = cv.imread('CATS_DOGS/CATS_DOGS/train/CAT/4.jpg')


print(cat4.shape)
# cat4 = cv.cvtColor(cat4,cv.COLOR_BGR2RGB)
# plt.imshow(cat4)
# plt.show()
# cv.imshow(

dog =cv.imread('CATS_DOGS/CATS_DOGS/train/DOG/2.jpg')
# dog = cv.cvtColor(dog,cv.COLOR_BGR2RGB)
print(dog.shape)

# need to data in same set

from  tensorflow.keras.preprocessing.image import ImageDataGenerator
image_gen =ImageDataGenerator(rotation_range=30,
                              width_shift_range=0.1,
                              height_shift_range=0.1,
                              rescale=1/255,
                              shear_range=.2,
                              zoom_range=0.2,
                              horizontal_flip=True,
                              fill_mode='nearest')
# plt.imshow(image_gen.random_transform(dog))
# plt.show()
image_gen.flow_from_directory('CATS_DOGS/CATS_DOGS/test')

# input shape = 150,150,3

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation,Dropout,Flatten,Conv2D,MaxPool2D

model = Sequential()

model.add(Conv2D(filters=32,kernel_size=(3,3),input_shape=(150,150,3),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters=64,kernel_size=(3,3),input_shape=(150,150,3),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters=64,kernel_size=(3,3),input_shape=(150,150,3),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(128))
model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss= 'binary_crossentropy',
              optimizer = 'adam',
              metrics=['accuracy'])

model.summary()
batch_size = 16
train_image_gen = image_gen.flow_from_directory('CATS_DOGS/CATS_DOGS/train',
                                                target_size=(150,150),
                                                batch_size=batch_size,
                                                class_mode='binary')
test_image_gen = image_gen.flow_from_directory('CATS_DOGS/CATS_DOGS/test',
                                                target_size=(150,150),
                                                batch_size=batch_size,
                                                class_mode='binary')


print(train_image_gen.class_indices)
results =model.fit_generator(train_image_gen,epochs=100,steps_per_epoch=150,validation_data=test_image_gen,validation_steps=12)

print(results.history['accuracy'])
plt.plot(results.history['accuracy'])
plt.show()

# model.save('catAndDog50epochs.h5')



plt.show()
cv.waitKey(0)