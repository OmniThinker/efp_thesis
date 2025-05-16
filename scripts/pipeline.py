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
    EarlyStoppingCallback,
    DataCollatorWithPadding,
    AutoModelForSequenceClassification
)
from certainty import (
    load_file, CACHE_DIR, seed_everything, RANDOM_SEED,
    load_events, BIOWeightedLossTrainer, WeightedLossTrainer, calc_split, concat_trigger,
    encode_dataset, calculate_class_weights, extract_triggers
)
from datasets import Dataset
from torch.utils.data import DataLoader
import evaluate

precision = evaluate.load("precision")
recall = evaluate.load("recall")
f1 = evaluate.load("f1")


def get_pred_word_fact(parsed):
    word_fact = {}
    for sample in parsed:
        key = sample
        if key in word_fact:
            word_fact[key] += 1
        else:
            word_fact[key] = 1
    return word_fact


def get_true_word_fact(parsed):
    word_fact = {}
    for sample in parsed:
        key = sample[4]
        if key in word_fact:
            word_fact[key].append(sample)
        else:
            word_fact[key] = [sample]
    return word_fact


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


def compute_metrics_factuality(eval_pred):
    preds, labels = eval_pred
    predictions = np.argmax(preds, axis=1)
    return {"discovered_asserted_f1": f1.compute(predictions=predictions, references=labels, pos_label=1)['f1'],
            "discovered_asserted_precision": precision.compute(predictions=predictions, references=labels, pos_label=1)['precision'],
            "discovered_asserted_recall": recall.compute(predictions=predictions, references=labels, pos_label=1)['recall'],
            "discovered_other_f1": f1.compute(predictions=predictions, references=labels, pos_label=0)['f1'],
            "discovered_other_precision": precision.compute(predictions=predictions, references=labels, pos_label=0)['precision'],
            "discovered_other_recall": recall.compute(predictions=predictions, references=labels, pos_label=0)['recall'], }


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

    print("Loading training data")
    train = load_file('en_train.json')
    print("Loading dev data")
    test = load_file(test_file)

    print("Dropping duplicates")
    train = pd.DataFrame(train).drop_duplicates('text').drop_duplicates('events').to_dict("records")
    test = pd.DataFrame(test).drop_duplicates('text').drop_duplicates('events').to_dict("records")

    print("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR, local_files_only=True, trust_remote_code=True)

    label_list = []
    id2label = {}
    i = 0
    id2label[i] = "O"
    label_list.append("O")
    i += 1
    label_list.append("B-trigger")
    id2label[i] = "B-trigger"
    label_list.append("I-trigger")
    id2label[i + 1] = "I-trigger"
    i += 2

    label2id = {label: idx for idx, label in id2label.items()}

    print("Encoding train")
    train_encoded = encode_dataset(train, tokenizer, label2id, False)

    test_encoded = encode_dataset(test, tokenizer, label2id, False)
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
        count = label_counts.get(label_id, 1)
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
                        {"true": el[0],
                         "pred": el[0],
                         "trigger": key,
                         "label": "spurious"
                         }
                        for el in value
                    ]
                elif len(value) > len(true_wf[key]):
                    # Key is in true, but length is longer for pred, we then have more spurious events
                    spurious += [
                        {"true": el[0],
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
                         "pred": el[0],
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
                         "pred": t[0],
                         "polarity": t[1],
                         "genericity": t[2],
                         "type": t[3],
                         "trigger": key,
                         "label": "undiscovered",
                         "text": t[4]}
                        for t in value
                    ]
        print("Constructing DF")
        trigger_fp = len(spurious)
        trigger_fn = len(undiscovered)
        trigger_tp = len(discovered)
        trigger_precision = trigger_tp / (trigger_tp + trigger_fp) if (trigger_tp + trigger_fp) > 0 else 0.0
        trigger_recall = trigger_tp / (trigger_tp + trigger_fn) if (trigger_tp + trigger_fn) > 0 else 0.0
        trigger_f1 = (2 * trigger_precision * trigger_recall) / (trigger_precision + trigger_recall) if (trigger_precision + trigger_recall) > 0 else 0.0

        return {
            "trigger_precision": trigger_precision,
            "trigger_recall": trigger_recall,
            "trigger_f1": trigger_f1,
        }

    OUTPUT_DIR = "../models/pipeline" + str(random)
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
        num_train_epochs=epochs,  # epochs,
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
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )

    trainer.train()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    model.eval()
    subset = test_dataset
    subset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    dataloader = DataLoader(subset, batch_size=batch_size)
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            all_logits.append(logits.cpu())
            all_labels.append(batch["labels"].cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    preds = [[id2label[label.item()] for label in labels] for labels in np.argmax(all_logits, axis=2)]

    parsed_true = []
    for sample in test:
        parsed_true.append([(
            event['event_modality'],
            event['event_polarity'],
            event['event_genericity'],
            event['event_type'],
            event['trigger'][0][0]) for event in sample["events"]])

    samples = []
    for j, ex, pred in zip(range(len(preds)), test_input_ids, preds):
        tokens = tokenizer.convert_ids_to_tokens(ex, skip_special_tokens=True)
        pred = pred[:len(tokens)]
        current_trigger = []
        events = []
        for i, p in enumerate(pred):
            if p == 'B-trigger':
                if current_trigger:
                    trigger_word = tokenizer.convert_tokens_to_string(current_trigger).strip()
                    events.append(trigger_word)
                current_trigger = [tokens[i]]
            elif p == 'I-trigger':
                current_trigger.append(tokens[i])
            else:
                if current_trigger:
                    trigger_word = tokenizer.convert_tokens_to_string(current_trigger).strip()
                    events.append(trigger_word)
                    current_trigger = []
                continue
        if current_trigger:
            trigger_word = tokenizer.convert_tokens_to_string(current_trigger).strip()
            events.append(trigger_word)

        actual = {
            "text": test[j]['text'],
            "true": parsed_true[j],
            "pred": events
        }

        samples.append(actual)

    for sample in samples:
        pred_facts = get_pred_word_fact(sample['pred'])
        true_facts = get_true_word_fact(sample['true'])
        discovered = []
        spurious = []
        undiscovered = []
        for key in pred_facts.keys():
            for i in range(pred_facts[key]):
                if key in true_facts:
                    te = true_facts[key].pop(0)
                    discovered.append(te)
                    if len(true_facts[key]) == 0:
                        true_facts.pop(key, None)
                else:
                    spurious.append(key)
            if key in true_facts:
                undiscovered += true_facts[key]
        for key in true_facts.keys():
            undiscovered += true_facts[key]

        sample['discovered'] = discovered
        sample['undiscovered'] = undiscovered
        sample['spurious'] = spurious

    # Factuality Classifier
    fact_id2label = {0: "Other", 1: "Asserted"}
    fact_label2id = {"Other": 0, "Asserted": 1}

    train_events, test_events, _ = load_events('en_train.json', test_file, 'en_test.json')

    tokenizer_fact = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR, local_files_only=True)
    factuality_train_set = Dataset.from_pandas(pd.DataFrame(train_events)[["label", "text", "trigger", "type", "genericity", "polarity", "modality"]])
    factuality_test_set = Dataset.from_pandas(pd.DataFrame(test_events)[["label", "text", "trigger", "type", "genericity", "polarity", "modality"]])

    factuality_train_set = factuality_train_set.map(concat_trigger, batched=False).map(lambda data: tokenizer(data["features"], truncation=True), batched=True)
    factuality_test_set = factuality_test_set.map(concat_trigger, batched=False).map(lambda data: tokenizer(data["features"], truncation=True), batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_labels = torch.tensor(factuality_train_set['label'])
    class_weights = calculate_class_weights(train_labels)
    model_fact = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2, id2label=fact_id2label, label2id=fact_label2id
    )

    seed_everything(random)

    training_args_fact = TrainingArguments(
        output_dir="../models/factuality" + str(random),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        report_to="wandb",
        load_best_model_at_end=True,
        lr_scheduler_type='reduce_lr_on_plateau',
        save_total_limit=1,
        learning_rate=0.000005,
        warmup_ratio=0.1,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
    )

    trainer_fact = WeightedLossTrainer(
        model=model_fact,
        args=training_args_fact,
        train_dataset=factuality_train_set,
        eval_dataset=factuality_test_set,
        tokenizer=tokenizer_fact,
        data_collator=DataCollatorWithPadding(tokenizer_fact),
        compute_metrics=compute_metrics_factuality,
        class_weights=class_weights,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )

    trainer_fact.train()
    spurious = []
    discovered = []
    undiscovered = []

    for sample in samples:
        for disc in sample['discovered']:
            modality, polarity, genericity, t, trigger = disc
            text = sample['text']
            discovered.append({
                "modality": modality,
                "polarity": polarity,
                "genericity": genericity,
                "trigger": trigger,
                "text": text,
                "type": t,
                "label": 1 if modality == 'Asserted' else 0
            })

        for spur in sample['spurious']:
            text = sample['text']
            spurious.append({
                "modality": "Asserted",
                "polarity": "UNKNOWN",
                "genericity": "UNKNOWN",
                "trigger": spur,
                "text": text,
                "type": "UNKNOWN",
                "label": 1
            })
        for undisc in sample['undiscovered']:
            modality, polarity, genericity, t, trigger = undisc
            text = sample['text']
            undiscovered.append({
                "pred": "Asserted" if modality == 'Other' else "Other",
                "modality": modality,
                "polarity": polarity,
                "genericity": genericity,
                "trigger": trigger,
                "text": text,
                "type": t,
                "label": 1 if modality == 'Asserted' else 0
            })

    discovered_set = Dataset.from_pandas(pd.DataFrame(discovered, columns=["label", "text", "trigger", "type", "genericity", "polarity", "modality"])[
                                         ["label", "text", "trigger", "type", "genericity", "polarity", "modality"]])
    print(discovered_set)

    spurious_set = Dataset.from_pandas(pd.DataFrame(spurious, columns=["label", "text", "trigger", "type", "genericity", "polarity", "modality"])[
                                       ["label", "text", "trigger", "type", "genericity", "polarity", "modality"]])
    print(spurious_set)
    undiscovered_set = Dataset.from_pandas(pd.DataFrame(undiscovered, columns=["label", "text", "trigger", "type", "genericity", "polarity", "modality"])[
                                           ["label", "text", "trigger", "type", "genericity", "polarity", "modality"]])
    print(undiscovered_set)

    discovered_set = discovered_set.map(concat_trigger, batched=False).map(lambda data: tokenizer(data["features"], truncation=True), batched=True)
    spurious_set = spurious_set.map(concat_trigger, batched=False).map(lambda data: tokenizer(data["features"], truncation=True), batched=True)
    undiscovered_set = undiscovered_set.map(concat_trigger, batched=False).map(lambda data: tokenizer(data["features"], truncation=True), batched=True)

    discovered_pred = torch.tensor([])
    discovered_true = torch.tensor([])
    spurious_pred = torch.tensor([])
    spurious_true = torch.tensor([])
    undiscovered_pred = torch.tensor([])
    undiscovered_true = torch.tensor([])

    if len(discovered_set) > 0:
        x, y = calc_split(discovered_set, model_fact, data_collator, batch_size, device)
        discovered_pred = np.argmax(x, axis=1)
        discovered_true = y

    if len(spurious_set) > 0:
        x, y = calc_split(spurious_set, model_fact, data_collator, batch_size, device)
        spurious_pred = np.argmax(x, axis=1)
        spurious_true = 1 - spurious_pred

    if len(undiscovered_set) > 0:
        x, y = calc_split(undiscovered_set, model_fact, data_collator, batch_size, device)
        undiscovered_pred = 1 - y
        undiscovered_true = y

    labels = torch.cat((discovered_true, spurious_true, undiscovered_true))
    predictions = torch.cat((discovered_pred, spurious_pred, undiscovered_pred))
    metrics = {"eval/asserted_f1": f1.compute(predictions=predictions, references=labels, pos_label=1)['f1'],
               "eval/asserted_precision": precision.compute(predictions=predictions, references=labels, pos_label=1)['precision'],
               "eval/asserted_recall": recall.compute(predictions=predictions, references=labels, pos_label=1)['recall'],
               "eval/other_f1": f1.compute(predictions=predictions, references=labels, pos_label=0)['f1'],
               "eval/other_precision": precision.compute(predictions=predictions, references=labels, pos_label=0)['precision'],
               "eval/other_recall": recall.compute(predictions=predictions, references=labels, pos_label=0)['recall'],
               "eval/discovered_asserted_f1": f1.compute(predictions=discovered_pred, references=discovered_true, pos_label=1)['f1'],
               "eval/discovered_asserted_precision": precision.compute(predictions=discovered_pred, references=discovered_true, pos_label=1)['precision'],
               "eval/discovered_asserted_recall": recall.compute(predictions=discovered_pred, references=discovered_true, pos_label=1)['recall'],
               "eval/discovered_other_f1": f1.compute(predictions=discovered_pred, references=discovered_true, pos_label=0)['f1'],
               "eval/discovered_other_precision": precision.compute(predictions=discovered_pred, references=discovered_true, pos_label=0)['precision'],
               "eval/discovered_other_recall": recall.compute(predictions=discovered_pred, references=discovered_true, pos_label=0)['recall'], }

    wandb.log(metrics)
    for i, disc in enumerate(discovered):
        disc['pred'] = fact_id2label[discovered_pred[i].item()]
        disc['true'] = fact_id2label[discovered_true[i].item()]
        disc['label'] = 'discovered'

    for i, spur in enumerate(spurious):
        spur['pred'] = fact_id2label[spurious_pred[i].item()]
        spur['true'] = fact_id2label[spurious_true[i].item()]
        spur['label'] = 'spurious'

    for i, undisc in enumerate(undiscovered):
        undisc['pred'] = fact_id2label[undiscovered_pred[i].item()]
        undisc['true'] = fact_id2label[undiscovered_pred[i].item()]
        undisc['label'] = 'undiscovered'

    df = pd.concat([pd.DataFrame(discovered), pd.DataFrame(spurious), pd.DataFrame(undiscovered)])
    m = model_name.split("/")[-1]
    df.to_csv(f"../results/pipeline/{csv_name}.csv")


if __name__ == "__main__":
    args = parse_args()
    main(args)
