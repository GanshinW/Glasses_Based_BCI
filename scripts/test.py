# scripts/test.py
import os
from utils.load_npz import load_dataset, print_dataset_stats

def main():
    # 1) find the project root (one level up from scripts/)
    project_root = os.path.dirname(os.path.dirname(__file__))
    # 2) build the path to chewing_dataset.npz under simulation/
    dataset_file = os.path.join(project_root, 'simulation', 'chewing_dataset.npz')

    # load + print stats
    dataset = load_dataset(dataset_file)
    from models.multimodal_model import MultiModalNet

    # instantiate:
    model = MultiModalNet(
        n_channels=8,
        n_samples=int(fs * duration),
        n_bands=len(bands),
        img_out_dim=64,
        hidden_dim=64,
        n_classes=5,      # e.g., Calm/Pos/Neg/Eye/Jaw
        use_img=True      # or False to disable the image branch
    )

if __name__ == '__main__':
    main()
    