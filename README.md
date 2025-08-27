# Glasses_Based_BCI
The workflow:
acquire/intent_acquisition.py / emotion_acquisition.py → collect signals into CSV
acquire/dataset_builder.py → convert into NPZ dataset
scripts/intent_pipeline.ipynb / emotion_pipeline.ipynb → preprocessing, feature extraction, multimodal training, saving models (PTH)
scripts/realtime_bci_predictor.py → run real-time prediction