import os
import torch
from sklearn.model_selection import train_test_split


class DataPreprocessor:
    def __init__(self, data_path: str = "./data.txt"):
        if not os.path.exists(data_path.strip()):
            raise FileNotFoundError("Path to data doesn't exist.")

        if not data_path.endswith(".txt"):
            raise ValueError("The file is not a text file. Only text files are supported.")

        with open(data_path, 'r') as file:
            self.data = file.read().splitlines()

        self._prep_conversion()

    def split_data(self, train_split: float, dev_test_split: float):
        train_words, temp_words = train_test_split(self.data, train_size=train_split, random_state=42)
        dev_words, test_words = train_test_split(temp_words, test_size=dev_test_split, random_state=42)

        return self._prep_xs_ys(train_words), self._prep_xs_ys(dev_words), self._prep_xs_ys(test_words)

    def _prep_conversion(self):
        chars = sorted(list(set(''.join(self.data))))

        self.char_to_int = {s: i + 1 for i, s in enumerate(chars)}
        self.char_to_int['.'] = 0
        # int_to_char seems to be redundant. See if you can remove it safely
        self.int_to_char = {i: s for s, i in self.char_to_int.items()}

        two_chars = set()
        for c1 in chars + ["."]:
            for c2 in chars + ["."]:
                two_chars.add(c1 + c2)

        two_chars = sorted(list(two_chars))

        self.char2_to_int = {s: i for i, s in enumerate(two_chars)}
        self.int_to_char2 = {i: s for i, s in enumerate(two_chars)}

    def _prep_xs_ys(self, word_set: list[str]):
        xs = torch.tensor([], dtype=torch.long)
        ys = torch.tensor([], dtype=torch.long)

        for w in word_set:
            chs = ["."] + list(w) + ["."]
            for ch1, ch2, ch3 in zip(chs, chs[1:], chs[2:]):
                ix1 = self.char2_to_int[ch1 + ch2]
                ix2 = self.char_to_int[ch3]
                xs = torch.cat((xs, torch.tensor([ix1])))
                ys = torch.cat((ys, torch.tensor([ix2])))

        return xs, ys
