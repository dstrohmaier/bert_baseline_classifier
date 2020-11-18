import pandas as pd
import logging
import torch

from typing import Tuple, List
from transformers import BertTokenizer
from torch.utils.data import TensorDataset


class DataReader(object):
    def __init__(self,
                 model_name: str,
                 hyper_dict: dict,
                 label_dict: dict,
                 label_type=torch.long):

        self.hyper_dict = hyper_dict
        self.label_dict = label_dict
        self.tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=True)
        self.label_type = label_type  # needs to be torch.float for regression

    def tokenize_encode(self, text: str) -> List[int]:
        tokens = self.tokenizer.tokenize(text)
        encoded_tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        return encoded_tokens

    @staticmethod
    def pad_sequences(sequences: list, max_len: int) -> torch.Tensor:
        out_dims = (len(sequences), max_len)
        out_tensor = torch.zeros(*out_dims, dtype=torch.long)

        for i, tensor in enumerate(sequences):
            length = tensor.size(0)
            if length <= max_len:
                out_tensor[i, :length, ...] = tensor
            else:
                out_tensor[i, :length, ...] = tensor[:max_len]

        return out_tensor

    def read_data(self, df: pd.DataFrame,
                  load_columns: Tuple[str, str],
                  target_column: str = ""
                  ) -> TensorDataset:
        input_ids = []
        token_type_ids = []

        count_overlong = 0
        for sentence1, sentence2 in zip(df[load_columns[0]].values, df[load_columns[1]].values):

            assert type(sentence1) == str,  f"Sentence1 not string: {sentence1} | {sentence2}"
            assert type(sentence2) == str, f"Sentence2 not string: {sentence1} | {sentence2}"

            first_ids = self.tokenize_encode("[CLS] " + sentence1 + " [SEP]")
            first_types = [0 for _ in first_ids]
            second_ids = self.tokenize_encode(sentence2 + " [SEP]")
            second_types = [1 for _ in second_ids]

            if len(first_ids+second_ids) > self.hyper_dict["max_len"]:
                count_overlong += 1

            input_ids.append(torch.tensor(first_ids + second_ids))
            token_type_ids.append(torch.tensor(first_types + second_types))

        logging.info(f"Found {count_overlong} definition pairs exceeding max_len")

        input_ids = self.pad_sequences(input_ids, self.hyper_dict["max_len"])
        token_type_ids = self.pad_sequences(token_type_ids, self.hyper_dict["max_len"])

        attention_masks = []

        for seq in input_ids:
            seq_mask = [float(i > 0) for i in seq]
            attention_masks.append(seq_mask)

        attention_masks = torch.tensor(attention_masks, dtype=torch.long)

        assert type(input_ids) == torch.Tensor, f"Wrong type: {type(input_ids)} for input_ids. Should be Tensor"
        assert type(token_type_ids) == torch.Tensor, \
            f"Wrong type: {type(token_type_ids)} for token_type_ids. Should be Tensor"
        assert type(attention_masks) == torch.Tensor, \
            f"Wrong type: {type(attention_masks)} for attention_masks. Should be Tensor"

        if target_column:
            labels = [self.label_dict[label] for label in df.label.values]
            labels = torch.tensor(labels, dtype=self.label_type)
            tensor_data = TensorDataset(input_ids, token_type_ids, attention_masks, labels)
        else:
            tensor_data = TensorDataset(input_ids, token_type_ids, attention_masks)

        assert df.shape[0] == input_ids.shape[0], "length of DataFrame and length of input do not match"

        return tensor_data


def create_dfs_tuple(path, names, num_splits=5):
    train_dfs = []
    valid_dfs = []

    for i in range(num_splits):
        train_dfs.append(pd.read_csv(path + f"{i}_train.tsv",
                                     delimiter='\t',
                                     encoding="UTF-8",
                                     names=names,
                                     header=None,
                                     keep_default_na=False))
        valid_dfs.append(pd.read_csv(path + f"{i}_valid.tsv",
                                     delimiter='\t',
                                     encoding="UTF-8",
                                     names=names,
                                     header=None,
                                     keep_default_na=False))

    return tuple(train_dfs), tuple(valid_dfs)