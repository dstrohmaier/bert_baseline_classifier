import random
import logging

from typing import List

from utilities.logging_utils import log_dict


SELECTED_HYPERDICTS = [
    {
        "batch_size": 5,
        "decay_rate": 0.01,
        "epochs": 20,
        "learning_rate": 1e-05,
        "max_grad_norm": 1,
        "max_len": 130,
        "warmup_steps": 50
    }
]

STANDARD_SPACE = {
    "max_len_space":  list(range(90, 301, 10)),
    "batch_size_space": list(range(8, 65, 8)),
    "decay_rate_space": [i / 1000 for i in range(0, 15, 5)],
    "epochs_space": list(range(3, 100)),
    "learning_rate_space": [1e-4, 2e-5, 1e-5, 9e-6, 5e-6, 1e-6],
    "max_grad_norm_space": [1],
    "warmup_steps_space": list(range(80, 120, 10))
}


def draw_hyperparameters(space_dict: dict) -> dict:
    log_dict("space dict", space_dict)

    suffix_len = len("_space")
    hyper_dict = {key[:-suffix_len]: random.choice(value) for key, value in space_dict.items()}

    return hyper_dict


def create_hyperdicts_list(num_draws: int = 10) -> List[dict]:
    hyperdicts_list = []

    for draw in range(num_draws):
        logging.info(f"Validation run: {draw}")

        hyper_dict = draw_hyperparameters(STANDARD_SPACE)

        while hyper_dict in hyperdicts_list:
            hyper_dict = draw_hyperparameters(STANDARD_SPACE)
        hyperdicts_list.append(hyper_dict)

    return hyperdicts_list
