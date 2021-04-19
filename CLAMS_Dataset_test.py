import pandas as pd
import torch
import unittest

from CLAMS_Dataset import CLAMS_Dataset
from transformers import XLMRobertaTokenizer


class TestCLAMSDataset(unittest.TestCase):
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
        self.CLAMS_dataset = CLAMS_Dataset(
            self.dataset, self.tokenizer, self.max_seq_len
        )

    def test_len(self):
        ## TODO: Test that the length of self.boolq_dataset is correct.
        ## len(self.boolq_dataset) should equal len(self.dataset).
        self.assertEqual(len(self.CLAMS_dataset), len(self.dataset))

    def test_item(self):
        ## TODO: Test that, for each element of self.boolq_dataset, 
        ## the output of __getitem__ (accessible via self.boolq_dataset[idx])
        ## has the correct keys, value dimensions, and value types.
        ## Each item should have keys ["input_ids", "attention_mask", "labels"].
        ## The input_ids and attention_mask values should both have length self.max_seq_len
        ## and type torch.long. The labels value should be a single numeric value.
        for i in range(len(self.CLAMS_dataset)):
            self.assertEqual(list(self.CLAMS_dataset[i].keys()), ["input_ids", "attention_mask", "labels"])
            self.assertEqual(self.CLAMS_dataset[i]["input_ids"].dtype, torch.long)
            self.assertEqual(self.CLAMS_dataset[i]["attention_mask"].dtype, torch.long)
            self.assertEqual(type(self.CLAMS_dataset[i]["labels"]), int)


if __name__ == "__main__":
    unittest.main()
