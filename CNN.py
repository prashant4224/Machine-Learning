# 1. Thêm các thư viện cần thiết
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_val, y_val = x_train[50000:60000,:], y_train[50000:60000]
x_train, y_train = x_train[:50000,:], y_train[:50000]

print(x_train.shape)

x_train = x_train.reshape(x_train.shape[0], 28 ,28 ,1)
x_val = x_val.reshape(x_val.shape[0], 28 ,28 ,1)
x_test = x_test.reshape(x_test.shape[0], 28 ,28 ,1)


# 4. One hot encoding label (Y)
Y_train = np_utils.to_categorical(y_train, 10)
Y_val = np_utils.to_categorical(y_val, 10)
Y_test = np_utils.to_categorical(y_test, 10)
print('Dữ liệu y ban đầu ', y_train[0])
print('Dữ liệu y sau one-hot encoding ',Y_train[0])

model = Sequential()
model.add(Conv2D(32, (3 ,3 ), activation = 'sigmoid' , input_shape = (28, 28 , 1)))
model.add(Conv2D(32, (3 ,3 ), activation = 'sigmoid'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Flatten())
model.add(Dense( 128, activation= 'sigmoid'))
model.add(Dense( 10, activation= 'softmax'))

model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics=['accuracy'])

H = model.fit(x_train, Y_train, validation_data=(x_val, Y_val),
          batch_size=100, epochs=10, verbose=1)


# 9. Đánh giá model với dữ liệu test set
score = model.evaluate(x_test, Y_test, verbose=0)
print(score)


# 10. Dự đoán ảnh
plt.imshow(x_test[0].reshape(28,28), cmap='gray')

y_predict = model.predict(x_test[0].reshape(1,28,28,1))
print('Giá trị dự đoán: ', np.argmax(y_predict))


