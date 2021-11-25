import pandas as pd
import argparse
import os
import random
import pickle


random.seed(1985)

parser = argparse.ArgumentParser()
parser.add_argument('--data-root',
                    default='/Users/nkroher/Projects/COFLA/',
                    dest='data_root')
args = parser.parse_args()

data_root = args.data_root
if not os.path.isdir(data_root):
    print('Data root not found at', data_root)

data_files = []

meta_path = os.path.join(data_root, 'meta.csv')
meta = pd.read_csv(meta_path, delimiter=';')
for k, v in meta.groupby(['FAMILY']):
    print(k, len(v))


for k, row in enumerate(meta.iterrows()):
    idx, data = row
    audio_path = os.path.join(data_root, data['File_path'][1:])
    pickle_path = os.path.splitext(audio_path)[0] + '.p'
    if not os.path.isfile(pickle_path):
        continue
    data_files.append(pickle_path)

random.shuffle(data_files)
split_idx = int(0.9 * len(data_files))
train_files = data_files[:split_idx]
val_files = data_files[split_idx:]

pickle.dump(train_files, open('train_files.p', 'wb'))
pickle.dump(val_files, open('val_files.p', 'wb'))
