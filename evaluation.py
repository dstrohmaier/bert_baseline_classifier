from sklearn.metrics import f1_score


def average_evaluation(dict_list: list) -> dict:
    all_keys = dict_list[0].keys()
    averaged_dict = {key: sum(d[key] for d in dict_list)/len(dict_list) for key in all_keys}

    return averaged_dict


def evaluate_from_list(correct_labels: list, predictions: list) -> dict:
    assert len(predictions) == len(correct_labels), "predictions and correct labels differ in length"

    accuracy = sum(1 if pred == cor else 0 for pred, cor in zip(predictions, correct_labels)) / len(predictions)

    unique_labels = set(correct_labels)
    macro_f1 = f1_score(correct_labels, predictions, average="macro")

    overall_measures = {
        "accuracy": accuracy,
        "macro_f1": macro_f1
    }

    label_measures = {
        f"macro_f1_{label}": f1_score(correct_labels, predictions, labels=[label], average="macro") for
        label in unique_labels
    }

    return {**overall_measures, **label_measures}
