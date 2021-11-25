from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dropout, MaxPooling2D, Reshape, LSTM, Bidirectional, Flatten, Dense, Concatenate, AveragePooling2D
from tensorflow.keras.optimizers import Adam
import random
import numpy as np
import pickle


NUM_MEL_BINS = 128
NUM_TONALITY_CQT = 12
NUM_TIME_STEPS = 862  # 10s audio @ fs=44.1kHz and hop_size=512
NUM_CLASSES = 9


def timbre_rhythm_network(mel_input):
    # based on VGG-ish model from https://github.com/DTaoo/VGGish/blob/master/vggish.py
    # some layers commented to reduce parameter count
    # paper: https://arxiv.org/abs/1609.09430

    # Block 1
    x = Conv2D(16, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv1')(mel_input)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool1')(x)
    x = Dropout(0.2)(x)

    # Block 2
    x = Conv2D(32, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool2')(x)
    x = Dropout(0.2)(x)

    # Block 3
    x = Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv3/conv3_1')(x)
    # x = Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv3/conv3_2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool3')(x)
    x = Dropout(0.2)(x)

    # Block 4
    x = Conv2D(128, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv4/conv4_1')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool4')(x)
    x = AveragePooling2D((1, 54))(x)
    x = Flatten(name='flatten_')(x)
    return x


def tonality_network(cqt_input):
    # simple feed forward network
    x = Dense(64, activation='relu')(cqt_input)
    return x


def flamenco_net():
    cqt_input = Input(shape=(2 * NUM_TONALITY_CQT))
    mel_input = Input(shape=(NUM_MEL_BINS, NUM_TIME_STEPS, 1))
    tonality_out = tonality_network(cqt_input)
    rhythm_timbre_out = timbre_rhythm_network(mel_input)
    concat = Concatenate()([tonality_out, rhythm_timbre_out])
    dense = Dense(128, activation='relu')(concat)
    dense = Dense(32, activation='relu')(dense)
    dense = Dense(NUM_CLASSES, activation='softmax')(dense)
    model = Model([mel_input, cqt_input], dense)
    model.compile(optimizer=Adam(lr=0.001),
                  loss='categorical_crossentropy',
                  metrics=['acc'])
    return model


def data_generator(file_list, batch_size, num_steps, shuffle):
    while True:
        if shuffle:
            random.shuffle(file_list)
        for s in range(num_steps):
            X_mel = np.zeros((batch_size, NUM_MEL_BINS, NUM_TIME_STEPS, 1))
            X_chroma = np.zeros((batch_size, 2 * NUM_TONALITY_CQT))
            y = np.zeros((batch_size, NUM_CLASSES))
            for b in range(batch_size):
                data = pickle.load(open(file_list[s * batch_size + b], 'rb'))
                mel = data['mel']
                mel_mean = np.mean(mel, axis=1)
                mel = (mel.T - mel_mean.T).T
                chroma = data['chorma']
                label = data['label']
                song_len = mel.shape[1]
                if shuffle:
                    s_idx = random.randint(0, song_len - NUM_TIME_STEPS - 1)
                else:
                    s_idx = int(0.5 * song_len) + int(0.5 * NUM_TIME_STEPS)
                    s_idx = min([s_idx, song_len - NUM_TIME_STEPS - 1])
                X_mel[b, :, :, 0] = mel[:, s_idx: s_idx + NUM_TIME_STEPS]
                X_chroma[b, :NUM_TONALITY_CQT] = np.mean(chroma, axis=1)
                X_chroma[b, NUM_TONALITY_CQT:] = np.std(chroma, axis=1)
                y[b, label] = 1
            yield [X_mel, X_chroma], y


if __name__ == '__main__':
    network = flamenco_net()
    network.summary()
