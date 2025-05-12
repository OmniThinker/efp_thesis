#!/usr/bin/env pythonimport sns
import wandb
from wandb import Table
from datasets import Dataset
import evaluate
import sys
import os
import torch
from transformers import TrainerCallback, TrainerControl, TrainerState
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, Trainer
import numpy as np
import argparse
import torch
import torch.nn as nn
import pandas as pd

sys.path.append(os.path.join(os.path.abspath(os.getcwd()), ".."))

from certainty import id2label, label2id, load_events, TRAIN_FILENAME, TEST_FILENAME, DEV_FILENAME, RANDOM_SEED, CACHE_DIR, seed_everything

OUTPUT_DIR = "../models/bert_baseline"

accuracy = evaluate.load("accuracy")
precision = evaluate.load("precision")
recall = evaluate.load("recall")
f1 = evaluate.load("f1")


class WeightedLossTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")

        outputs = model(**inputs)

        logits = outputs.logits

        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits, labels)

        if return_outputs:
            return loss, outputs
        return loss


def calculate_class_weights(labels):
    class_counts = torch.bincount(labels)
    total_samples = labels.size(0)
    weights = total_samples / (len(class_counts) * class_counts)
    return weights.to("cuda")  # Move to GPU if applicable


def gen_preprocess_function(tokenizer: AutoTokenizer):
    return lambda data: tokenizer(data["features"], truncation=True)


def concat_trigger(data: Dataset):
    data["features"] = "[" + data["trigger"] + "]" + ": " + data["text"]
    return data


def concat_type(data: Dataset):
    data["features"] = "[" + data["type"] + ", " + data["trigger"] + "]" + ": " + data["text"]
    return data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concat", type=str, default="trigger", choices=["trigger", "type", "polarity", "genericity"])
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--decay", type=float, default=0.01)
    parser.add_argument("--random", type=int, default=RANDOM_SEED)
    parser.add_argument("--model", type=str, default='google-bert/bert-base-uncased')
    parser.add_argument("--name", type=str, default='bert-baseline-using-trigger')
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--frozen", type=str, default=False)

    args = parser.parse_args()
    return args


def main(args):
    seed_everything(args.random)

    train_events, dev_events, test_events = load_events(TRAIN_FILENAME, DEV_FILENAME, TEST_FILENAME)

    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=CACHE_DIR, local_files_only=True, trust_remote_code=True)
    preprocess_function = gen_preprocess_function(tokenizer)

    train_set = Dataset.from_pandas(pd.DataFrame(train_events)[["label", "text", "trigger", "type", "genericity", "polarity", "modality"]])
    dev_set = Dataset.from_pandas(pd.DataFrame(dev_events)[["label", "text", "trigger", "type", "genericity", "polarity", "modality"]])
    test_set = Dataset.from_pandas(pd.DataFrame(test_events)[["label", "text", "trigger", "type", "genericity", "polarity", "modality"]])

    eval_set = test_set

    if args.concat == "trigger":
        gen_features = concat_trigger
    else:
        gen_features = concat_type

    train_set = train_set.map(gen_features, batched=False).map(preprocess_function, batched=True)
    dev_set = dev_set.map(gen_features, batched=False).map(preprocess_function, batched=True)
    test_set = test_set.map(gen_features, batched=False).map(preprocess_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, cache_dir=CACHE_DIR, local_files_only=True, num_labels=2, id2label=id2label, label2id=label2id, trust_remote_code=True
    )
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print("DEVICE " + device)

    train_labels = torch.tensor(train_set['label'])

    class_weights = calculate_class_weights(train_labels)

    if args.frozen:
        for param in model.base_model.parameters():
            param.requires_grad = False

    def compute_metrics(eval_pred, prefix=""):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        metrics = {
            "other_precision": precision.compute(predictions=predictions, references=labels, pos_label=0)["precision"],
            "other_recall": recall.compute(predictions=predictions, references=labels, pos_label=0)["recall"],
            "other_f1": f1.compute(predictions=predictions, references=labels, pos_label=0)["f1"],
            "asserted_precision": precision.compute(predictions=predictions, references=labels, pos_label=1)["precision"],
            "asserted_recall": recall.compute(predictions=predictions, references=labels, pos_label=1)["recall"],
            "asserted_f1": f1.compute(predictions=predictions, references=labels, pos_label=1)["f1"],
        }
        results = []
        for i, (pred, lab) in enumerate(zip(predictions, labels)):
            sample = eval_set[i]
            results.append({
                "true": lab,
                "pred": pred,
                "polarity": sample['polarity'],
                "genericity": sample['genericity'],
                "type": sample['type'],
                "trigger": sample['trigger'],
                "text": sample['text']
            })
        df = pd.DataFrame(results)
        concat = "trigger" if args.concat == "trigger" else "type"

        df.to_csv('../results/TEST_baseline_' + concat + '_' + str(args.random) + ".csv")

        return metrics

    name = args.name + ":" + args.model + ":" + str(args.lr) + ":" + str(args.frozen) + ":" + args.concat + ":" + str(args.random)
    warmup_ratio = 0.1

    wandb.init(
        project="certainty",
        name=name,
        config={
            "concat_feature": args.concat,
            "lr": args.lr,
            "epochs": args.epochs,
            "decay": args.decay,
            "random_state": args.random,
            "model": args.model,
            "trigger": args.concat,
            "frozen": args.frozen,
            "warmup_ratio": warmup_ratio
        },
        dir="..",
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR + "/" + args.model,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.decay,
        warmup_ratio=warmup_ratio,
        evaluation_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        report_to="wandb",
        lr_scheduler_type="reduce_lr_on_plateau",
        save_strategy="no",
    )

    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=test_set,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        class_weights=class_weights
    )

    trainer.train()
    wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    main(args)
