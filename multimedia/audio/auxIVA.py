from scipy.io import wavfile
import pyroomacoustics as pra
import numpy as np 
# read multichannel wav file
# audio.shape == (nsamples, nchannels)
fs, audio = wavfile.read("multi_sp_SOBI.wav")

audio = audio[:, :3]
# STFT analysis parameters
fft_size = 4096  # `fft_size / fs` should be ~RT60
hop = fft_size // 2  # half-overlap
win_a = pra.hann(fft_size)  # analysis window
# optimal synthesis window
win_s = pra.transform.compute_synthesis_window(win_a, hop)

# STFT
# X.shape == (nframes, nfrequencies, nchannels)
X = pra.transform.analysis(audio, fft_size, hop, win=win_a)

# Separation
Y = pra.bss.auxiva(X, n_iter=2000)

# iSTFT (introduces an offset of `hop` samples)
# y contains the time domain separated signals
# y.shape == (new_nsamples, nchannels)
y = pra.transform.synthesis(Y, fft_size, hop, win=win_s)

import ipdb; ipdb.set_trace()
for m in range(audio.shape[1]):
    wavfile.write("multi_sp_K={}_auxIVA.wav".format(m+1), fs, y[:, m].astype(np.int16))