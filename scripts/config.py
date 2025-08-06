# config.py
class Config:
    # Data
    sampling_rate = 250
    trial_duration = 3.0
    bands = [(4, 8), (8, 13), (13, 30)]
    
    # Model
    hidden_dim = 64
    img_out_dim = 64
    use_img = True
    
    # Training
    batch_size = 16
    learning_rate = 1e-3
    num_epochs = 50
    patience = 10
    
    # System
    num_workers = 0 