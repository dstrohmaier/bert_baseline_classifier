import os
import numpy as np
import pandas as pd

from typing import Tuple


def split_df(df: pd.DataFrame, column: str = "sentence1",  num_splits: int = 5) -> Tuple[list, list]:
    unique_words = df[column].unique()
    np.random.shuffle(unique_words)
    word_splits = np.array_split(unique_words, num_splits)
    train_dfs = [df[~df[column].isin(selected)] for selected in word_splits]
    valid_dfs = [df[df[column].isin(selected)] for selected in word_splits]

    assert len(train_dfs) == len(valid_dfs)
    assert len(train_dfs) == num_splits
    assert sum(v_df.shape[0] for v_df in valid_dfs) == df.shape[0]

    return train_dfs, valid_dfs


def split_and_save(data: pd.DataFrame, output_directory: str):
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)

    train_dfs, valid_dfs = split_df(data)

    for counter, (t_df, v_df) in enumerate(zip(train_dfs, valid_dfs)):

        train_out_path = output_directory + f"{counter}_train.tsv"
        valid_out_path = output_directory + f"{counter}_valid.tsv"

        t_df.to_csv(train_out_path, sep="\t", encoding="UTF-8", index=False, header=False)
        v_df.to_csv(valid_out_path, sep="\t", encoding="UTF-8", index=False, header=False)


if __name__ == '__main__':
    single_df = pd.read_csv("data/train.tsv", delimiter='\t', encoding="UTF-8")
    split_and_save(single_df, "data/split/")
