import librosa
import numpy as np
import matplotlib.pyplot as plt


FS = 44100
NUM_TONALITY_CQT = 60


def load_audio(_audio_path):
    _samples, _ = librosa.load(_audio_path,
                               sr=FS)
    _samples = _samples / np.max(np.abs(_samples))
    return _samples


def extract_mel(_samples):
    _mel = librosa.feature.melspectrogram(_samples,
                                          sr=FS,
                                          hop_length=512,
                                          fmin=60.,
                                          fmax=12000.,
                                          n_mels=128)
    _mel = librosa.amplitude_to_db(np.abs(_mel), ref=0.0)
    return _mel


def extract_chroma(_samples):
    _chroma = librosa.feature.chroma_cqt(_samples,
                                         sr=FS,
                                         fmin=librosa.note_to_hz('C2'),
                                         hop_length=512)
    template_maj = [0.748, 0.060, 0.488, 0.082, 0.67, 0.46, 0.096, 0.715, 0.104, 0.366, 0.057, 0.4]
    #template_fm = [0.0445651, 0.06235872, 0.06158099, 0.1403152, 0.07885851, 0.07613546, 0.06867356, 0.083829,
    #               0.08598009, 0.04263247, 0.1652883, 0.0897826]
    template_fm = [0.1652883, 0.0897826, 0.0445651, 0.06235872, 0.06158099, 0.1403152, 0.07885851, 0.07613546, 0.06867356, 0.083829,
                   0.08598009, 0.04263247]
    mean_chroma = np.mean(_chroma, axis=1)
    maj_scores = np.zeros((11, 1))
    fm_scores = np.zeros((11, 1))
    for shift in range(0, 11):
        h2_s = np.roll(mean_chroma, shift)
        maj_scores[shift] = np.corrcoef(template_maj, h2_s)[0][1]
        fm_scores[shift] = np.corrcoef(template_fm, h2_s)[0][1]
    maj_score = max(maj_scores)
    fm_score = max(fm_scores)
    if maj_score > fm_score:
        shift = np.argmax(maj_scores)
    else:
        shift = np.argmax(fm_scores)
    _chroma = np.roll(_chroma, shift, axis=0)
    return _chroma


if __name__ == '__main__':
    samples = load_audio('test.mp3')
    chroma = extract_chroma(samples)
    mel = extract_mel(samples)
    plt.subplot(211)
    plt.imshow(chroma, aspect='auto', origin='lower', interpolation='none')
    plt.title('Chroma')
    plt.colorbar()
    plt.subplot(212)
    plt.imshow(mel, aspect='auto', origin='lower', interpolation='none')
    plt.title('Mel Spectrogram')
    plt.colorbar()
    plt.show()
