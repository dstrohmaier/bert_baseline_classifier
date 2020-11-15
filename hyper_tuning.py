import os
import json
import logging

from typing import Generator, Tuple, List  # shouldn't be needed after Python 3.9

from core import LabelClassifier
from evaluation import evaluate_from_list, average_evaluation
from utilities.logging_utils import log_dict
from utilities.load_data import DataReader


def cross_validate(model_name: str,
                   hyper_dict: dict,
                   label_dict: dict,
                   df_generator: Generator,
                   load_columns: Tuple[str, str],
                   target_column: str) -> dict:

    assert type(hyper_dict) == dict

    logging.info("Starting validation for hyperparameters:")
    log_dict("hyperparameters", hyper_dict)

    reader = DataReader(model_name, hyper_dict, label_dict)

    result_dict_list = []

    for train_df, validation_df in df_generator:
        classifier = LabelClassifier(model_name, hyper_dict, label_dict)
        steps_per_epoch = train_df.shape[0] // hyper_dict["batch_size"]

        t_dataset = reader.read_data(train_df, load_columns, target_column)
        v_dataset = reader.read_data(validation_df, load_columns)

        classifier.train_model(t_dataset, steps_per_epoch)
        predictions = classifier.eval_model(v_dataset)
        pred_df = validation_df.assign(**{target_column: predictions})

        predicted_labels = pred_df[target_column].tolist()
        correct_labels = pred_df[target_column].tolist()

        result_dict = evaluate_from_list(correct_labels, predicted_labels)
        logging.info("Preliminary result from one round in cross-validation")
        log_dict("results", result_dict)

        result_dict_list.append(result_dict)
        del classifier

    return average_evaluation(result_dict_list)


def search_parameter_list(model_name: str,
                          hyperdicts_list: List[dict],
                          label_dict: dict,
                          df_generator: Generator,
                          load_columns: Tuple[str, str],
                          target_column: str,
                          output_directory: str,
                          identifier: float) -> None:

    assert output_directory[-1] == "/", "output_directory has to end with slash"

    json_directory = output_directory + "json/"

    for directory in (output_directory, json_directory):
        if not os.path.isdir(directory):
            os.makedirs(directory)

    for i, hyper_dict in enumerate(hyperdicts_list):
        averaged_result_dict = cross_validate(model_name=model_name,
                                              hyper_dict=hyper_dict,
                                              label_dict=label_dict,
                                              df_generator=df_generator,
                                              load_columns=load_columns,
                                              target_column=target_column)

        logging.info("Average result for one hyperparameter setting")
        log_dict("averaged results", averaged_result_dict)

        draw_data = (identifier, hyper_dict, averaged_result_dict)

        file_path = json_directory + f"{identifier}_selection_{i}.json"

        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(draw_data, file, ensure_ascii=False, indent=4)
