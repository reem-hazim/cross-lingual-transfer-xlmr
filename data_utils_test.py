import data_utils_2 as data_utils
import pandas as pd
import torch
import unittest

from transformers import RobertaTokenizerFast

class TestDataUtils(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        self.dataset = pd.DataFrame.from_dict(
            {
                "sentence": ["the author that the guards like laughs", "the author that the guards like laugh", "the author that the guards like swims", "the author that the guards like swim"],
                "label": [True, False, True, False],
            }
        )
        self.max_seq_len = 4

    def test_encode_data(self):
        ## TODO: Write a unit test that asserts that the dimensions and dtype of the
        ## output of encode_data() are correct.
        ## input_ids should have shape [len(self.dataset), self.max_seq_len] and type torch.long.
        ## attention_mask should have the same shape and type.
        input_ids, attention_mask = data_utils.encode_data(self.dataset, self.tokenizer, self.max_seq_len)
        print(input_ids)
        print(attention_mask)

    def test_extract_labels(self):
        ## TODO: Write a unit test that asserts that extract_labels() outputs the
        ## correct labels, e.g. [1, 0].
        labels = data_utils.extract_labels(self.dataset)
        self.assertEqual(labels, [1, 0])

if __name__ == "__main__":
    unittest.main()