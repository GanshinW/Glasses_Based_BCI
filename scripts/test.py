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

if __name__ == '__main__':
    main()