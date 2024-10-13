from sklearn import utils
from Functions_and_Declarations import *
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras

DATA_PATH = os.path.join('MP_TRAINING_DATA')
actions = np.array(['hello', 'my', 'name','null'])
no_sequences = 200
sequence_length = 30



label_map = {label:num for num, label in enumerate(actions)}

sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

x = np.array(sequences)
y = keras.utils.to_categorical(labels).astype(int)
       

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
