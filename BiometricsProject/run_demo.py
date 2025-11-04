from src.dataset_creator import create_sample_signatures
from src.utils import create_sample_dataset


def run_demo():
    """Run a quick demo of the system"""
    print("Setting up demo...")

    # Create directories and samples
    create_sample_dataset()
    create_sample_signatures()

    print("\nDemo setup complete!")
    print("Now run 'main.py' to start using the system.")
    print("\nSample signatures created in 'data/samples/'")


if __name__ == "__main__":
    run_demo()