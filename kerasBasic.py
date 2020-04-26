import numpy as np
from numpy import genfromtxt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

data = genfromtxt('data/bank_note_data.txt',delimiter=',')
# print(data)
labels = data[:,4]
# print(labels)
features =data[:,0:4]
X= features
y=labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# print(len(X_train))
scaler_object =MinMaxScaler()
print(scaler_object.fit(X_train))
scaled_X_train = scaler_object.transform(X_train)
scaled_X_test = scaler_object.transform(X_test)

print(scaled_X_train.max())

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

model.add(Dense(4,input_dim=4,activation='relu'))

model.add(Dense(8,activation='relu'))

model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

print(model.fit(scaled_X_train,y_train,epochs=50,verbose=2))
model.predict_classes(scaled_X_test)

model.metrics_names
from sklearn.metrics import confusion_matrix,classification_report
prediction = model.predict_classes(scaled_X_test)
confusion_matrix(y_test,prediction)
print(classification_report(y_test,prediction)) #show the confusion metrics table
