def install_data():
    import os
    import subprocess
    import zipfile

    KAGGLE_DIR = os.path.expanduser("~/.kaggle")
    ZIP_PATH = "pothole-image-segmentation-dataset.zip"
    DATASET_DIR = "dataset"

    # Create ~/.kaggle directory
    os.makedirs(KAGGLE_DIR, exist_ok=True)

    # Download dataset
    subprocess.run(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            "farzadnekouei/pothole-image-segmentation-dataset",
        ],
        check=True,
    )

    # Extract dataset
    os.makedirs(DATASET_DIR, exist_ok=True)

    with zipfile.ZipFile(ZIP_PATH, "r") as zip_file:
        zip_file.extractall(DATASET_DIR)

    print(f"Dataset extracted to: {DATASET_DIR}")


if __name__ == "__main__":
    install_data()
