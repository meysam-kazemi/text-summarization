import os
import sys
from datasets import load_dataset
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from utils import read_config


class DataLoader:
    def __init__(self, config, language_code):
        """
        Initialize the DataLoader with the dataset name, language code, and local path.
        :param language_code: Language code ("es" for Spanish, "en" for English)
        """
        self.dataset_name = config["dataset"]['dataset_name']
        self.local_data_path = config["dataset"]['local_data_path']
        self.language_code = language_code

    def load_data(self):
        """
        Load data from local storage if available, otherwise download it.

        :return: Dataset object containing the loaded data
        """
        if os.path.exists(self.local_data_path):
            print(f"Loading data from {self.local_data_path}...")
            return self._load_local_data()
        else:
            print(f"Data not found locally. Downloading {self.dataset_name} ({self.language_code})...")
            dataset = self._download_data()
            self._save_local_data(dataset)
            return dataset

    def _load_local_data(self):
        """
        Load the dataset from local storage.

        :return: Dataset object containing the loaded data
        """
        return load_dataset(self.local_data_path)

    def _download_data(self):
        """
        Download data from the specified dataset and language code.

        :return: Dataset object containing the downloaded data
        """
        return load_dataset(self.dataset_name, self.language_code)

    def _save_local_data(self, dataset):
        """
        Save the dataset to local storage.

        :param dataset: Dataset object to save
        """
        # Save the dataset to local storage (this can be customized as needed)
        dataset.save_to_disk(self.local_data_path)
        print(f"Data downloaded and saved to {self.local_data_path}.")

# Example usage:
if __name__ == "__main__":
    # Define parameters
    config = read_config()
    spanish_code = "es"
    english_code = "en"
    
    # Define local paths
    spanish_local_path = "data/raw/spanish_dataset"
    english_local_path = "data/raw/english_dataset"

    # Create instances of DataLoader
    spanish_data_loader = DataLoader(config=config, spanish_code)
    english_data_loader = DataLoader(config=config, english_code)
    
    # Load the datasets
    spanish_dataset = spanish_data_loader.load_data()
    english_dataset = english_data_loader.load_data()
    
    # Display the first few entries of the English dataset
    print(english_dataset)