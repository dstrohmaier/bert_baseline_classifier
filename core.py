import numpy as np
import logging
import torch

from typing import List

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, BertForSequenceClassification, get_linear_schedule_with_warmup


class BaseClassifier(object):
    def __init__(self, model_name: str, hyper_dict: dict, label_dict: dict, device_name: str = "cuda"):
        self.device = torch.device(device_name)
        self.device_name = device_name

        self.hyper_dict = hyper_dict

        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(label_dict))
        self.model.cuda(self.device_name)

        self.label_dict = label_dict

    def train_model(self, encoded_data: TensorDataset, steps_per_epoch: int):
        logging.info(f"starting to train model")

        sampler = RandomSampler(encoded_data)
        dataloader = DataLoader(encoded_data, sampler=sampler, batch_size=self.hyper_dict["batch_size"])

        num_total_steps = steps_per_epoch * self.hyper_dict["epochs"]

        param_optimizer = list(self.model.named_parameters())
        no_decay = ["bias", "gamma", "beta", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': self.hyper_dict["decay_rate"]},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hyper_dict["learning_rate"], correct_bias=False)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.hyper_dict["warmup_steps"],
                                                    num_training_steps=num_total_steps)

        self.model.train()

        for _ in range(self.hyper_dict["epochs"]):
            logging.info("starting epoch")

            training_loss = []
            for batch in dataloader:
                batch = tuple(t.to(self.device) for t in batch)
                b_input_ids, b_type_ids, b_input_mask, b_labels = batch

                optimizer.zero_grad()
                outputs = self.model(b_input_ids, token_type_ids=b_type_ids, attention_mask=b_input_mask)
                logits = outputs[0]

                loss = self.calculate_loss(logits, b_labels)

                training_loss.append(loss.item())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.hyper_dict["max_grad_norm"])  # ordering?
                optimizer.step()
                scheduler.step()  # ordering assumes  pytorch >= 1.1.0

            logging.info(f"Training loss: {sum(training_loss) / len(training_loss)}")

    def calculate_loss(self, logits: torch.Tensor, b_labels: torch.Tensor):
        raise NotImplementedError

    def eval_model(self, encoded_data: TensorDataset) -> List:
        logging.info(f"starting to test model")

        sampler = SequentialSampler(encoded_data)
        dataloader = DataLoader(encoded_data, sampler=sampler, batch_size=self.hyper_dict["batch_size"])

        self.model.eval()

        predictions = []

        for batch in dataloader:
            batch = tuple(t.to(self.device) for t in batch)
            b_input_ids, b_type_ids, b_input_mask = batch

            with torch.no_grad():
                output = self.model(b_input_ids, token_type_ids=b_type_ids, attention_mask=b_input_mask)[0]

            b_predictions = self.extract_predictions(output)

            predictions.extend(b_predictions)

        return predictions

    def extract_predictions(self, output: torch.Tensor):
        raise NotImplementedError


class LabelClassifier(BaseClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert len(self.label_dict) > 1, "Only one label. RegressionClassifier should be used instead"

        self.inv_label_dict = {value: key for key, value in self.label_dict.items()}

    def convert_labels(self, label_tensor: torch.Tensor):
        converted_labels = []
        base = [0] * len(self.label_dict)
        for label in label_tensor:
            converted_labels.append(base[:label] + [1] + base[label + 1:])

        return torch.tensor(converted_labels, dtype=torch.float, device=self.device)

    def calculate_loss(self, logits: torch.Tensor, b_labels: torch.Tensor):
        converted_labels = self.convert_labels(b_labels)

        loss_fct = torch.nn.CrossEntropyLoss()
        loss_fct.to(self.device)
        loss = loss_fct(
            logits, converted_labels
        )
        return loss

    def extract_predictions(self, output: torch.Tensor):
        logits = output.detach().cpu().numpy()
        b_predictions = np.argmax(logits, axis=1).flatten()
        b_predictions = [self.inv_label_dict[int(prediction)] for prediction in b_predictions]
        return b_predictions


class WeightedLabelClassifier(LabelClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert "pos_weight" in self.hyper_dict.keys(), "WeightedLabelClassifier called without weights."

    def calculate_loss(self, logits: torch.Tensor, b_labels: torch.Tensor):
        pos_weight = torch.tensor(self.hyper_dict["pos_weight"], dtype=torch.long, device=self.device)

        converted_labels = self.convert_labels(b_labels)

        loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        loss_fct.to(self.device)
        loss = loss_fct(
            logits, converted_labels
        )

        return loss


class RegressionClassifier(BaseClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert len(self.label_dict) == 1, "RegressionClassifier called with more than one label"

    def calculate_loss(self, logits: torch.Tensor, b_labels: torch.Tensor):
        scores = torch.sigmoid(logits)

        loss_fct = torch.nn.L1Loss()
        loss_fct.to(self.device)
        loss = loss_fct(
            scores, b_labels
        )

        return loss

    def extract_predictions(self, output: torch.Tensor):
        scores = torch.sigmoid(output)
        b_predictions = scores.detach().cpu().numpy()

        return b_predictions.tolist()
