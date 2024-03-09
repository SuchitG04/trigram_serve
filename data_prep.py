import os
from sklearn.model_selection import train_test_split


class DataPreprocessor:
    def __init__(self, data_path: str = "./data.txt"):
        if not os.path.exists(data_path.strip()):
            raise FileNotFoundError("Path to data doesn't exist.")

        if not data_path.endswith(".txt"):
            raise ValueError("The file is not a text file. Only text files are supported.")

        with open(data_path, 'r') as file:
            self.data = file.read().splitlines()

    def split_data(self, train_split: float, dev_test_split: float):
        train_words, temp_words = train_test_split(self.data, train_size=train_split, random_state=42)
        dev_words, test_words = train_test_split(temp_words, test_size=dev_test_split, random_state=42)

        return train_words, dev_words, test_words
