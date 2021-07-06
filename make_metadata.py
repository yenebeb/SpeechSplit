import sys
sys.path.append('../Real-Time-Voice-Cloning/')

import os
import pickle
import numpy as np
from pathlib import Path

# Real Time VOice Cloning encoder
from encoder import inference as encoder

# make spect f0 imports
import soundfile as sf
from scipy import signal
from librosa.filters import mel
from numpy.random import RandomState
from pysptk import sptk
from utils import butter_highpass
from utils import speaker_normalization
from utils import pySTFT

# Root directory voices
rootDir = '../LibriSpeech/train-clean-100'

# Target directory mel and f0
targetDir_f0 = 'assets/raptf0'
targetDir = 'assets/spmel'

dirName, subdirList, _ = next(os.walk(rootDir))
print('Found directory: %s' % dirName)

# Settings spect/f0
#####
mel_basis = mel(16000, 1024, fmin=90, fmax=7600, n_mels=80).T
min_level = np.exp(-100 / 20 * np.log(10))
b, a = butter_highpass(30, 16000, order=5)
# min/max fundamental frequency taken from original code and combined in 1
lo, hi = 50, 600
#####

num_uttrs = 10 # Number of utterances per speaker

# Load encoder model (hardcoded for now)
encoder_path = Path("../Real-Time-Voice-Cloning/encoder/saved_models/pretrained.pt")
encoder.load_model(encoder_path)


speakers = []
# Access all directories in rootDir, where name of directory is speaker
for speaker in sorted(subdirList):
    print('Processing speaker: %s' % speaker)
    utterances = []
    utterances.append(speaker)

    # for train clean 100 we need to access next directory before accessing files:
    # _, lastDirList, _ = next(os.walk(os.path.join(dirName,speaker)))

    # Create path to speaker
    dirVoice = Path(os.path.join(dirName,speaker))
    # Get all wav files from speaker
    fileList = list(dirVoice.glob("**/*.flac"))
    
    # use hardcoded onehot embeddings in order to be cosistent with the test speakers
    # modify as needed
    # may use generalized speaker embedding for zero-shot conversion

    ## OLD 
    # spkid = np.zeros((82,), dtype=np.float32)
    # if speaker == 'p226':
    #     spkid[1] = 1.0
    # else:
    #     spkid[7] = 1.0
    # utterances.append(spkid)
    ##

    # Speaker embedding using GE2E encoder
    # make speaker embedding
    assert len(fileList) >= num_uttrs
    idx_uttrs = np.random.choice(len(fileList), size=num_uttrs, replace=False)
    embs = []
    fileNameSaves = []
    prng = RandomState(int(os.path.basename(os.path.dirname(fileList[0]))[1:])) 
    for i in range(num_uttrs):
        fileName = str(os.path.basename(fileList[idx_uttrs[i]]))
        if not os.path.exists(os.path.join(targetDir, speaker)):
            os.makedirs(os.path.join(targetDir, speaker))
        if not os.path.exists(os.path.join(targetDir_f0, speaker)):
            os.makedirs(os.path.join(targetDir_f0, speaker))    
        # preproces and generate embedding
        preprocessed_wav = encoder.preprocess_wav(fileList[idx_uttrs[i]])
        embed = encoder.embed_utterance(preprocessed_wav)
        embs.append(embed)
    
        

        # read audio file
        x, fs = sf.read(fileList[idx_uttrs[i]])
        assert fs == 16000
        if x.shape[0] % 256 == 0:
            x = np.concatenate((x, np.array([1e-06])), axis=0)
        y = signal.filtfilt(b, a, x)
        wav = y * 0.96 + (prng.rand(y.shape[0])-0.5)*1e-06

        # compute spectrogram
        D = pySTFT(wav).T
        D_mel = np.dot(D, mel_basis)
        D_db = 20 * np.log10(np.maximum(min_level, D_mel)) - 16
        S = (D_db + 100) / 100        

        # extract f0
        f0_rapt = sptk.rapt(wav.astype(np.float32)*32768, fs, 256, min=lo, max=hi, otype=2)
        index_nonzero = (f0_rapt != -1e10)
        mean_f0, std_f0 = np.mean(f0_rapt[index_nonzero]), np.std(f0_rapt[index_nonzero])
        f0_norm = speaker_normalization(f0_rapt, index_nonzero, mean_f0, std_f0)

        assert len(S) == len(f0_rapt)
            
        np.save(os.path.join(targetDir, speaker, fileName[:-4]),
                S.astype(np.float32), allow_pickle=False)    
        np.save(os.path.join(targetDir_f0, speaker, fileName[:-4]),
                f0_norm.astype(np.float32), allow_pickle=False)
        fileNameSaves.append(os.path.join(speaker,fileName[:-4]))
    utterances.append(np.mean(embs, axis=0))

    
    # create file list
    # for filePath in sorted(fileList):
    #     fileName = str(os.path.basename(filePath))
    utterances.append(fileNameSaves)
    speakers.append(utterances)
    
with open(os.path.join(targetDir, 'train.pkl'), 'wb') as handle:
    pickle.dump(speakers, handle)