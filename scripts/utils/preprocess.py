# utils/preprocess.py
import numpy as np
from scipy.signal import iirnotch, filtfilt, butter, sosfiltfilt
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import FastICA
import pywt

_EPS = 1e-8

def notch_filter(x, fs, f0=50.0, q=25.0):
    # x: (..., T)
    b, a = iirnotch(w0=f0/(fs/2.0), Q=q)
    return filtfilt(b, a, x, axis=-1)

def bandpass_filter(x, fs, low=1.0, high=40.0, order=4):
    # x: (..., T)
    lowc = low/(fs/2.0); highc = high/(fs/2.0)
    sos = butter(order, [lowc, highc], btype='band', output='sos')
    return sosfiltfilt(sos, x, axis=-1)

def moving_average(x, k):
    # x: (..., T)
    if k is None or k <= 1: return x
    w = np.ones(k, dtype=np.float32)/float(k)
    return np.apply_along_axis(lambda s: np.convolve(s, w, mode='same'), -1, x)

def gaussian_smooth(x, sigma_s):
    # x: (..., T)
    if sigma_s is None or sigma_s <= 0: return x
    return gaussian_filter1d(x, sigma=sigma_s, axis=-1, mode='nearest')

def wavelet_denoise(x, name='db4', level=None, mode='soft'):
    # x: (..., T)
    def _denoise_1d(sig):
        coeffs = pywt.wavedec(sig, name, level=level)
        sigma = np.median(np.abs(coeffs[-1]))/0.6745 + _EPS
        thr = sigma*np.sqrt(2*np.log(sig.size))
        coeffs[1:] = [pywt.threshold(c, thr, mode=mode) for c in coeffs[1:]]
        return pywt.waverec(coeffs, name)[:sig.size]
    return np.apply_along_axis(_denoise_1d, -1, x)

def ica_remove(x, n_components=None, remove_idx=None):
    # x: (C, T) per trial
    C, T = x.shape
    n_comp = n_components if n_components is not None else C
    if n_comp <= 1: return x
    ica = FastICA(n_components=n_comp, whiten='unit-variance', random_state=42, max_iter=1000)
    S = ica.fit_transform(x.T)  # (T, n_comp)
    if remove_idx:
        S[:, np.array(remove_idx, dtype=int)] = 0.0
    X_hat = ica.inverse_transform(S).T  # (C, T)
    return X_hat.astype(np.float32)

def zscore(x):
    # x: (..., T)
    m = x.mean(axis=-1, keepdims=True)
    s = x.std(axis=-1, keepdims=True) + _EPS
    return (x - m)/s

def preprocess_batch(
    X, fs,
    notch_freq=50.0, notch_q=25.0,
    band=(1.0, 40.0), order=4,
    ma_window=None,
    gaussian_sigma=None,
    wavelet=None,           # e.g. dict(name='db4', level=None, mode='soft')
    ica_n=None, ica_remove_idx=None,
    do_zscore=True
):
    """
    X: (N, C, T), float
    fs: sampling rate
    return: X_proc (N, C, T), float32
    """
    Xp = X.astype(np.float32, copy=True)

    if notch_freq and notch_freq > 0:
        Xp = notch_filter(Xp, fs, f0=notch_freq, q=notch_q)
    if band is not None:
        low, high = band
        Xp = bandpass_filter(Xp, fs, low=low, high=high, order=order)
    if ma_window and ma_window > 1:
        Xp = moving_average(Xp, k=ma_window)
    if gaussian_sigma and gaussian_sigma > 0:
        Xp = gaussian_smooth(Xp, sigma_s=gaussian_sigma)
    if wavelet:
        Xp = wavelet_denoise(Xp, name=wavelet.get('name','db4'),
                                level=wavelet.get('level',None),
                                mode=wavelet.get('mode','soft'))
    if ica_n or (ica_remove_idx and len(ica_remove_idx)>0):
        out = np.empty_like(Xp)
        for i in range(Xp.shape[0]):
            out[i] = ica_remove(Xp[i], n_components=ica_n, remove_idx=ica_remove_idx or [])
        Xp = out
    if do_zscore:
        Xp = zscore(Xp)

    return Xp.astype(np.float32)
