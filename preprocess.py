import os
from src.data.WESAD.preprocess import main as preprocess_wesad

def main():
    # Make preprocessed directory
    os.makedirs("data/preprocessed", exist_ok=True)

    # Preprocess WESAD dataset
    preprocess_wesad()

if __name__ == "__main__":
    main()
