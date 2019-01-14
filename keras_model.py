from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

import numpy as np
import data_utils

data_dim = 128
timesteps = 128

train_data = data_utils.load_training_data()
test_data = data_utils.load_test_data()

#    loader = data_utils.DataLoader(data=data,batch_size=train_config.batch_size, num_steps=train_config.num_steps)
data_loader = data_utils.DataLoader(train_data, 7352, 1)
# x_test, y_test = data_utils.DataLoader(train_data, 128, 1).next_batch()


# expected input data shape: (batch_size, timesteps, data_dim)
model = Sequential()
model.add(LSTM(256, return_sequences=True,
               input_shape=(1, 1)))  # returns a sequence of vectors of dimension 32
model.add(LSTM(256, return_sequences=True))  # returns a sequence of vectors of dimension 32
model.add(LSTM(256, return_sequences=True))  # return a single vector of dimension 32
model.add(Dense(128, activation="sigmoid", name="DENSE1"))
model.add(Dense(72, activation="sigmoid", name="DENSE2"))

model.add(Dense(1, activation='softmax'))

model.compile(loss="mse",
              optimizer='adam',
              metrics=['accuracy'])

model.summary()
while data_loader.has_next():
    x_train, y_train = data_loader.next_batch()
    print(x_train.shape)
    x_train = x_train.reshape((128, 1, 1))
    y_train = y_train.reshape((128, 1, 1))
    print(x_train.shape)
    print(y_train.shape)
    model.fit(x_train, y_train, batch_size=24, epochs=1, validation_split=0.2)
