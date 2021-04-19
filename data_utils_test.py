import data_utils
import pandas as pd
import torch
import unittest

from transformers import XLMRobertaTokenizer

class TestDataUtils(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
        self.dataset = pd.DataFrame.from_dict(
            {
                "sentence": ["the pilots the dancers hate smiles", "no farmers that the skaters like have ever been popular"],
                "idx": [0, 1],
                "label": [False, True],
            }
        )
        self.max_seq_len = 4

    def test_sample(self):
        ## An example of a basic unit test, using class variables initialized in
        ## setUpClass().
        self.assertEqual(self.max_seq_len, 4)

    def test_encode_data(self):
        ## TODO: Write a unit test that asserts that the dimensions and dtype of the
        ## output of encode_data() are correct.
        ## input_ids should have shape [len(self.dataset), self.max_seq_len] and type torch.long.
        ## attention_mask should have the same shape and type.
        input_ids, attention_mask = data_utils.encode_data(self.dataset, self.tokenizer, self.max_seq_len)
        print(input_ids)
    def test_extract_labels(self):
        ## TODO: Write a unit test that asserts that extract_labels() outputs the
        ## correct labels, e.g. [1, 0].
        labels = data_utils.extract_labels(self.dataset)
        self.assertEqual(labels, [0, 1])

if __name__ == "__main__":
    unittest.main()
