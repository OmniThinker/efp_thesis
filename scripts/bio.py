import sys
import os
sys.path.append(os.path.join(os.path.abspath(os.getcwd()), ".."))
import pandas as pd
import numpy as np
import wandb
import argparse

from collections import Counter
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from certainty import (
    load_file, CACHE_DIR, seed_everything, RANDOM_SEED,
    BIOWeightedLossTrainer, encode_dataset, extract_triggers
)
from datasets import Dataset


def get_word_fact(parsed, is_true):
    word_fact = {}
    for sample in parsed:
        factuality = sample[0]
        word = sample[1]
        if is_true:
            polarity = sample[2]
            genericity = sample[3]
            e_type = sample[4]
            text = sample[5]
        if word in word_fact:
            if is_true:
                word_fact[word].append((factuality, polarity, genericity, e_type, text))
            else:
                word_fact[word].append((factuality,))
        else:
            if is_true:
                word_fact[word] = [(factuality, polarity, genericity, e_type, text)]
            else:
                word_fact[word] = [(factuality,)]
    return word_fact


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lr_strategy", type=str, default="reduce_lr_on_plateau")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--random", type=int, default=RANDOM_SEED)
    parser.add_argument("--model_name", type=str, default='google-bert/bert-base-uncased')
    parser.add_argument("--name", type=str, default='BERT_NER_modality_class')
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--scaling_factor", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--label_smoothing", type=float, default=0.01)
    parser.add_argument("--test_file", type=str, default="en_dev.json")
    parser.add_argument("--test", type=int, default=0)

    args = parser.parse_args()
    return args


def main(args):
    name = args.name
    lr = args.lr
    lr_strategy = args.lr_strategy
    epochs = args.epochs
    decay = args.decay
    random = args.random
    model_name = args.model_name
    batch_size = args.batch_size
    warmup_ratio = args.warmup_ratio
    scaling_factor = args.scaling_factor
    alpha = args.alpha
    label_smoothing = args.label_smoothing
    test_file = args.test_file
    test = args.test
    m = model_name.split("/")[-1]

    if test == 0:
        csv_name = f"weighted-{random}-{m}-{lr}-{lr_strategy}"
    else:
        csv_name = f"test-weighted-{random}-{m}-{lr}-{lr_strategy}"

    seed_everything(random)

    train = load_file('en_train.json')
    test = load_file(test_file)

    train = pd.DataFrame(train).drop_duplicates('text').drop_duplicates('events').to_dict("records")
    test = pd.DataFrame(test).drop_duplicates('text').drop_duplicates('events').to_dict("records")

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR, local_files_only=True, trust_remote_code=True)

    label_list = []
    id2label = {}
    i = 0
    id2label[i] = "O"
    label_list.append("O")
    i += 1
    label_list.append("B-Asserted")
    id2label[i] = "B-Asserted"
    label_list.append("I-Asserted")
    id2label[i + 1] = "I-Asserted"
    label_list.append("B-Other")
    id2label[i + 2] = "B-Other"
    label_list.append("I-Other")
    id2label[i + 3] = "I-Other"
    i += 4

    label2id = {label: idx for idx, label in id2label.items()}

    print("Encoding")
    train_encoded = encode_dataset(train, tokenizer, label2id, True)

    test_encoded = encode_dataset(test, tokenizer, label2id, True)
    test_input_ids = [ex["input_ids"].tolist() for ex in test_encoded]

    train_dataset = Dataset.from_list(train_encoded)
    test_dataset = Dataset.from_list(test_encoded)

    print("Loading model")
    model = AutoModelForTokenClassification.from_pretrained(
        model_name, cache_dir=CACHE_DIR, local_files_only=True, num_labels=len(id2label.items()), trust_remote_code=True, id2label=id2label, label2id=label2id
    )
    label_counts = Counter()

    for example in train_encoded:
        label_ids = example["labels"]
        label_ids = [int(label) for label in label_ids if label != -100]
        label_counts.update(label_ids)

    total_count = sum(label_counts.values())
    num_labels = len(label2id)

    scale = scaling_factor
    weights = [1.0] * num_labels
    for label_id in range(num_labels):
        count = label_counts.get(label_id, 1)  # avoid division by zero
        base = total_count / (num_labels * count)
        weights[label_id] = scale * (base ** alpha)

    class_weights = torch.tensor(weights, dtype=torch.float)

    def compute_metrics(p):
        preds, labels = p
        predictions = np.argmax(preds, axis=2)

        pred_ids = []
        for prediction, label in zip(predictions, labels):
            ids = []
            for pred_idx, label_idx in zip(prediction, label):
                if label_idx != -100:
                    ids.append(pred_idx.item())
            pred_ids.append(ids)

        label_preds = [
            [id2label[label_id] for label_id in seq]
            for seq in pred_ids
        ]

        parsed_pred = []

        for tokens_ids, labels_str in zip(test_input_ids, label_preds):
            tokens = tokenizer.convert_ids_to_tokens(tokens_ids, skip_special_tokens=True)
            print("Tokens: " + str(tokens))
            print("Label str:" + str(labels_str))
            pred_events = extract_triggers(tokens, labels_str, tokenizer)
            print("Pred events: " + str(pred_events))
            parsed_pred.append(pred_events)

        parsed_true = []
        for sample in test[:len(parsed_pred)]:
            parsed_true.append([(  # "Other" if event['event_genericity'] == "Generic" else event['event_modality'],
                event['event_modality'],
                event['trigger'][0][0],
                event['event_polarity'],
                event['event_genericity'],
                event['event_type'],
                sample['text']) for event in sample["events"]])

        print("Getting dicts")
        trues = [get_word_fact(sample, True) for sample in parsed_true]
        preds = [get_word_fact(sample, False) for sample in parsed_pred]

        print("Calculating metrics")
        spurious = []
        undiscovered = []
        discovered = []
        for true_wf, pred_wf in zip(trues, preds):
            for key, value in pred_wf.items():
                if key not in true_wf:
                    spurious += [
                        {"true": "Other" if el[0] == 'Asserted' else "Asserted",
                         "pred": el[0],
                         "trigger": key,
                         "label": "spurious"
                         }
                        for el in value
                    ]
                elif len(value) > len(true_wf[key]):
                    # Key is in true, but length is longer for pred, we then have more spurious events
                    spurious += [
                        {"true": "Other" if el[0] == 'Asserted' else "Asserted",
                         "pred": el[0],
                         "trigger": key,
                         "label": "spurious"
                         }
                        for el in value[len(true_wf[key]):]
                    ]
                else:
                    # key is in true, but length is shorter than for true, we have undiscovered events
                    undiscovered += [
                        {"true": el[0],
                         "pred": "Other" if el[0] == 'Asserted' else "Asserted",
                         "trigger": key,
                         "polarity": el[1],
                         "genericity": el[2],
                         "type": el[3],
                         "text": el[4],
                         "label": "undiscovered"
                         }
                        for el in true_wf[key][len(value):]
                    ]

            for key, value in true_wf.items():
                if key in pred_wf:
                    discovered += [
                        {"true": t[0],
                         "pred": p[0],
                         "polarity": t[1],
                         "genericity": t[2],
                         "type": t[3],
                         "trigger": key,
                         "label": "discovered",
                         "text": t[4]}
                        for t, p in zip(value, pred_wf[key])
                    ]
                else:
                    undiscovered += [
                        {"true": t[0],
                         "pred": "Other" if t[0] == 'Asserted' else "Asserted",
                         "polarity": t[1],
                         "genericity": t[2],
                         "type": t[3],
                         "trigger": key,
                         "label": "undiscovered",
                         "text": t[4]}
                        for t in value
                    ]
        print("Constructing DF")
        df = pd.DataFrame(spurious + discovered + undiscovered)
        trigger_fp = len(spurious)
        trigger_fn = len(undiscovered)
        trigger_tp = len(discovered)
        trigger_precision = trigger_tp / (trigger_tp + trigger_fp) if (trigger_tp + trigger_fp) > 0 else 0.0
        trigger_recall = trigger_tp / (trigger_tp + trigger_fn) if (trigger_tp + trigger_fn) > 0 else 0.0
        trigger_f1 = (2 * trigger_precision * trigger_recall) / (trigger_precision + trigger_recall) if (trigger_precision + trigger_recall) > 0 else 0.0

        discovered_other_fp = len(df[(df['label'] == 'discovered') & (df['true'] == 'Asserted') & (df['pred'] == 'Other')])
        discovered_other_fn = len(df[(df['label'] == 'discovered') & (df['true'] == 'Other') & (df['pred'] == 'Asserted')])
        discovered_other_tp = len(df[(df['label'] == 'discovered') & (df['true'] == 'Other') & (df['pred'] == 'Other')])

        discovered_asserted_fp = len(df[(df['label'] == 'discovered') & (df['true'] == 'Other') & (df['pred'] == 'Asserted')])
        discovered_asserted_fn = len(df[(df['label'] == 'discovered') & (df['true'] == 'Asserted') & (df['pred'] == 'Other')])
        discovered_asserted_tp = len(df[(df['label'] == 'discovered') & (df['true'] == 'Asserted') & (df['pred'] == 'Asserted')])

        discovered_other_precision = discovered_other_tp / (discovered_other_tp + discovered_other_fp) if (discovered_other_tp + discovered_other_fp) > 0 else 0.0
        discovered_other_recall = discovered_other_tp / (discovered_other_tp + discovered_other_fn) if (discovered_other_tp + discovered_other_fn) > 0 else 0.0
        discovered_other_f1 = (2 * discovered_other_precision * discovered_other_recall) / (discovered_other_recall +
                                                                                            discovered_other_precision) if (discovered_other_recall + discovered_other_precision) else 0.0

        discovered_asserted_precision = discovered_asserted_tp / (discovered_asserted_tp + discovered_asserted_fp) if (discovered_asserted_tp + discovered_asserted_fp) > 0 else 0.0
        discovered_asserted_recall = discovered_asserted_tp / (discovered_asserted_tp + discovered_asserted_fn) if (discovered_asserted_tp + discovered_asserted_fn) > 0 else 0.0
        discovered_asserted_f1 = (2 * discovered_asserted_precision * discovered_asserted_recall) / (discovered_asserted_recall +
                                                                                                     discovered_asserted_precision) if (discovered_asserted_recall + discovered_asserted_precision) > 0 else 0.0

        asserted_fp = len(df[(df['label'] == 'spurious') & (df['pred'] == 'Asserted')])
        asserted_fn = len(df[(df['label'] == 'undiscovered') & (df['true'] == 'Asserted')])
        other_fp = len(df[(df['label'] == 'spurious') & (df['pred'] == 'Other')])
        other_fn = len(df[(df['label'] == 'undiscovered') & (df['true'] == 'Other')])

        tot_as_fp = asserted_fp + discovered_asserted_fp
        tot_as_fn = asserted_fn + discovered_asserted_fn
        asserted_precision = discovered_asserted_tp / (discovered_asserted_tp + tot_as_fp) if (discovered_asserted_tp + tot_as_fp) > 0 else 0.0
        asserted_recall = discovered_asserted_tp / (discovered_asserted_tp + tot_as_fn) if (discovered_asserted_tp + tot_as_fn) > 0 else 0.0
        asserted_f1 = (2 * asserted_precision * asserted_recall) / (asserted_precision + asserted_recall) if (asserted_precision + asserted_recall) > 0 else 0.0

        tot_ot_fp = other_fp + discovered_other_fp
        tot_ot_fn = other_fn + discovered_other_fn
        other_precision = discovered_other_tp / (discovered_other_tp + tot_ot_fp) if (discovered_other_tp + tot_ot_fp) > 0 else 0.0
        other_recall = discovered_other_tp / (discovered_other_tp + tot_ot_fn) if (discovered_other_tp + tot_ot_fn) > 0 else 0.0
        other_f1 = (2 * other_precision * other_recall) / (other_precision + other_recall) if (other_precision + other_recall) > 0 else 0.0

        df.to_csv(f"../results/bio-factuality/{csv_name}.csv")

        return {
            "trigger_precision": trigger_precision,
            "trigger_recall": trigger_recall,
            "trigger_f1": trigger_f1,
            "discovered_other_recall": discovered_other_recall,
            "discovered_other_precision": discovered_other_precision,
            "discovered_other_f1": discovered_other_f1,
            "discovered_asserted_recall": discovered_asserted_recall,
            "discovered_asserted_precision": discovered_asserted_precision,
            "discovered_asserted_f1": discovered_asserted_f1,
            "asserted_precision": asserted_precision,
            "asserted_recall": asserted_recall,
            "asserted_f1": asserted_f1,
            "other_precision": other_precision,
            "other_recall": other_recall,
            "other_f1": other_f1,
        }

    OUTPUT_DIR = "../models/blabla"
    print("Initializing WANDB")
    wandb.init(
        project="certainty",
        name=name,
        config={
            "lr": lr,
            "lr_strategy": lr_strategy,
            "epochs": epochs,
            "decay": decay,
            "random_state": random,
            "model": model_name,
            "warmup_ratio": warmup_ratio,
            "scaling_factor": scaling_factor,
            "alpha": alpha,
            "label_smoothing": label_smoothing
        },
        dir="..",
    )

    print("Starting training")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=decay,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        report_to="wandb",
        lr_scheduler_type=lr_strategy,
        load_best_model_at_end=True,
        save_total_limit=1,
        metric_for_best_model="trigger_f1"
    )

    trainer = BIOWeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        label_smoothing=label_smoothing,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]  # Stop after 3 epochs without improvement
    )

    trainer.train()


if __name__ == "__main__":
    args = parse_args()
    main(args)
