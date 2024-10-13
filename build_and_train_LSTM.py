from keras import models, backend
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard
from preprocess_data import *
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score


log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))


optimizer = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

print("Learning rate before first fit:", model.optimizer.learning_rate.numpy())

model.load_weights('signs_04_200.h5')

try:
    epochs = 100
    #for e in range(epochs): 
        #m = model
        #model.fit(x_train, y_train, epochs=e+1, batch_size=1, callbacks=[tb_callback], shuffle=True, initial_epoch=e)
        #scores = model.evaluate(x_test, y_test, verbose=0)
        #print(f'Validation Score for epoch {e}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
    
except KeyboardInterrupt:
    pass    
#model.save('signs_04_200.h5')
#print('\n WEIGHTS SAVED!')


#yhat = model.predict(x_test)

#ytrue = np.argmax(y_test, axis=1).tolist()
#yhat = np.argmax(yhat, axis=1).tolist()

#           Detects true positives, true negatives, false positives, false negatives
#print(multilabel_confusion_matrix(ytrue, yhat))
#print(accuracy_score(ytrue, yhat))


