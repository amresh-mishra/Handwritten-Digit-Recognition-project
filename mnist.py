from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential #arranging the latyer in sequential order
from keras .layers import Conv2D, MaxPool2D, Flatten, Dense
import matplotlib.pyplot as plt

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1],X_train.shape[2],1))
X_test=X_test.reshape((X_test.shape[0],X_test.shape[1],X_test.shape[2],1))

plt.axis('off')
plt.imshow(X_test[45], cmap='gray')
plt.show()

X_train=X_train/25  #normalization of pixel value pixel value is in range 0 to 255 by deving max range
X_test=X_test/255

#One hot encoding target values
y_train=to_categorical(y_train)
y_test=to_categorical(y_test)


model=Sequential()
model.add(Conv2D(32,(3,3), activation ='relu', input_shape=(28,28,1)))    # , 32 is face ,3,3 is kernal size or filter, 1 for greyscale 
model.add(MaxPool2D(2,2))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(10, activation='softmax')) #output layer as multiclass classification softmax is used , output 0 to 9= 10


model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])


history=model.fit(X_train,y_train,validation_data = (X_test,y_test),epochs=5,batch_size=32)