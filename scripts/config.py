# scripts/config.py

class Config:
    # Data
    sampling_rate = 250
    trial_duration = 10.0
    bands = [(4, 8), (8, 13), (13, 30)]
    use_img = True  # enable time-frequency image branch

    # Model
    hidden_dim = 64
    img_out_dim = 64

    # Training
    batch_size = 16
    learning_rate = 1e-3
    num_epochs = 50
    patience = 10
    num_workers = 0

    # Preprocess (referencing methods 3–9)
    class Preprocess:
        enable = True
        # 3) Notch
        notch = True
        notch_freq = 50.0
        notch_q = 25.0
        # 4) Band-pass
        bandpass = True
        hp = 1.0
        lp = 40.0
        order = 4
        # 5) Smoothing
        moving_average = False
        ma_window = 0            # samples; 0=off
        gaussian = False
        gaussian_sigma_s = 0.0   # seconds; 0=off
        # 6) Wavelet denoise
        wavelet = False
        wavelet_name = 'db4'
        wavelet_level = None
        wavelet_mode = 'soft'
        # 7) ICA artifact removal
        ica = False
        ica_n_components = None
        ica_remove_idx = []      # e.g., [1,3]

    # Feature extraction params
    class Features:
        welch_nperseg = None
        welch_noverlap = None
        spec_nperseg = None
        spec_noverlap = None
        channel_method = 'average'   # 'average' | 'first' | 'pca'
