import argparse
import pickle
import time
import os
import pandas as pd
from features import extract_chroma, extract_mel, load_audio


parser = argparse.ArgumentParser()
parser.add_argument('--data-root',
                    default='/Users/nkroher/Projects/COFLA/',
                    dest='data_root')
args = parser.parse_args()

data_root = args.data_root
if not os.path.isdir(data_root):
    print('Data root not found at', data_root)
    exit(1)

meta_path = os.path.join(data_root, 'meta.csv')
meta = pd.read_csv(meta_path, delimiter=';')
meta = meta.sort_values(['Anthology', 'CD No', 'File_path'])
labels = ['BULERIA', 'CANTIÑA', 'FANDANGO', 'LEVANTE', 'MALAGUEÑA', 'SEGUIRIYA', 'SOLEÁ', 'TANGO', 'TONÁ']

for k, v in meta.groupby(['FAMILY']):
    print(k, len(v))

for k, row in enumerate(meta.iterrows()):
    t_s = time.time()
    idx, data = row
    audio_path = os.path.join(data_root, data['File_path'][1:])
    print('Extracting features for song ', k + 1, ' / ', len(meta), os.path.basename(audio_path))
    pickle_path = os.path.splitext(audio_path)[0] + '.p'
    #if os.path.isfile(pickle_path):
    #    print('Audio file already processed ', audio_path, ' - skipping!')
    #    continue
    if not os.path.isfile(audio_path):
        print('[ INFO ]: Audio file not found at ', audio_path, ' - skipping!')
        continue
    samples = load_audio(audio_path)
    chroma = extract_chroma(samples)
    mel = extract_mel(samples)
    data = {
        'chorma': chroma,
        'mel': mel,
        'label': labels.index(data['FAMILY'])
    }
    pickle.dump(data, open(pickle_path, 'wb'))
    print(' --> time elapsed: ', time.time() - t_s)
