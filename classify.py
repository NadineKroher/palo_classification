import pickle
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import math


NUM_MEL_BINS = 128
NUM_TONALITY_CQT = 12
NUM_TIME_STEPS = 862  # 10s audio @ fs=44.1kHz and hop_size=512

labels = ['BULERIA', 'CANTIÑA', 'FANDANGO', 'LEVANTE', 'MALAGUEÑA', 'SEGUIRIYA', 'SOLEÁ', 'TANGO', 'TONÁ']

model = load_model('palo_tag.h5')
val_files = pickle.load(open('val_files.p', 'rb'))
cm = np.zeros((9, 9))
num_correct = 0
print('Running evaluation on ', len(val_files), 'files')

for val_file in val_files:
    data = pickle.load(open(val_file, 'rb'))
    mel = data['mel']
    chroma = data['chorma']
    true_label = data['label']
    song_len = mel.shape[1]
    num_frames = int(math.floor(song_len / NUM_TIME_STEPS))
    X_mel = np.zeros((num_frames, NUM_MEL_BINS, NUM_TIME_STEPS, 1))
    X_chroma = np.zeros((num_frames, 2 * NUM_TONALITY_CQT))
    mel_mean = np.mean(mel, axis=1)
    mel = (mel.T - mel_mean.T).T
    chroma_mean = np.mean(chroma, axis=1)
    chroma_std = np.std(chroma, axis=1)
    for f in range(num_frames):
        s_idx = f * NUM_TIME_STEPS
        X_mel[f, :, :, 0] = mel[:, s_idx: s_idx + NUM_TIME_STEPS]
        X_chroma[f, :NUM_TONALITY_CQT] = chroma_mean
        X_chroma[f, NUM_TONALITY_CQT:] = chroma_std
    y_pred = np.mean(model.predict([X_mel, X_chroma]), axis=0)
    est_label = np.argmax(y_pred)
    if true_label == est_label:
        num_correct += 1
    cm[true_label, est_label] += 1

print('ACCURACY: ', 100 * num_correct / len(val_files))
fig, ax = plt.subplots()
# plt.imshow(cm, aspect='auto', origin='lower')
ax.matshow(cm, cmap=plt.cm.Blues)
plt.xticks(list(range(len(labels))), labels)
plt.xticks(rotation=45)
plt.ylabel('true label')
plt.xlabel('predicted label')
plt.yticks(list(range(len(labels))), labels)
for i in range(len(labels)):
    for j in range(len(labels)):
        c = cm[j, i]
        ax.text(i, j, str(c), va='center', ha='center')
plt.tight_layout()
plt.show()
