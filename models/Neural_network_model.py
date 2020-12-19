from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import ModelCheckpoint
import os
import numpy as np

OPTIMIZER = 'Adam'
BATCH_SIZE = 32 
EPOCHS = 100
LOSS = 'binary_crossentropy'
EVAL_METRIC = 'Accuracy'

class Model:
    ''' Model for neural networks with the  Keras library '''
    def __init__(self, X_train: object, X_test: object, y_train: object, y_test: object):
        self.X_train = X_train
        self.X_test = X_test
        self.y_test = y_test
        self.y_train = y_train
        
        self.model = Sequential()
        
    def define_model(self):
        self.model.add(Dense(30, kernel_regularizer='l2', activation='relu', input_shape=(29,)))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1, activation='sigmoid'))

    def train_model(self, save_path: str):
        self.model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=[EVAL_METRIC])
        checkpoint = ModelCheckpoint(filepath=os.path.join(save_path, 'best_model.h5'), verbose=0, save_best_only=True)
        self.history = self.model.fit(self.X_train, 
                                 self.y_train, 
                                 epochs=EPOCHS, 
                                 callbacks=[checkpoint], 
                                 validation_split=0.2,
                                 batch_size=BATCH_SIZE,
                                 shuffle=True,
                                 verbose=1
                                )
    
    def evaluate(self):
        loss, accuracy = self.model.evaluate(self.X_test, self.y_test)
        return loss, accuracy
    
    def predict(self) -> list:
        return self.model.predict_classes(self.X_test)
#        return np.argmax(self.model.predict(self.X_test), axis=-1)
    
    def get_metrics(self) -> object:
        return self.history
    
    def load_weights(self, name):
        self.model.load_weights(name)
