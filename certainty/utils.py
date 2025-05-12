from typing import Dict, Any, Sequence, Generator, List, get_args, Tuple
import os
import json
import pandas as pd
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import torch
import random
from torch.utils.data import DataLoader
# import seaborn as sns
from .types import EventSentence, EventType
from .constants import RANDOM_SEED
from datasets import Dataset

id2label = {0: "Other", 1: "Asserted"}
label2id = {"Other": 0, "Asserted": 1}


def load_file(filename: str) -> Sequence[Dict[str, Any]]:
    path = os.path.abspath(os.path.join("..", "data", "raw", "ace2005", filename))
    with open(path) as f:
        ds: Sequence[Dict[str, Any]] = json.load(f)
    return ds


def convert_events(ds: list[Dict[str, Any]]) -> Generator[EventSentence, None, None]:
    for sentence_dict in ds:
        for event in sentence_dict.get("events", []):
            yield {
                "sent_id": sentence_dict["sent_id"],
                "text": sentence_dict["text"],
                "type": event["event_type"],
                "modality": event["event_modality"],
                "label": 1 if event["event_modality"] == "Asserted" else 0,
                "polarity": event["event_polarity"],
                "genericity": event["event_genericity"],
                "trigger": event["trigger"][0][0],
                "trigger_idx": event["trigger"][1][0],
                "arguments": event["arguments"]
            }


def load_events(train_filename, dev_filename, test_filename) -> Tuple[List[EventSentence], List[EventSentence], List[EventSentence]]:
    ace_train: Sequence[Dict[str, Any]] = load_file(train_filename)
    ace_dev: Sequence[Dict[str, Any]] = load_file(dev_filename)
    ace_test: Sequence[Dict[str, Any]] = load_file(test_filename)

    train_df = pd.DataFrame(ace_train).drop_duplicates('text').drop_duplicates('events')
    dev_df = pd.DataFrame(ace_dev).drop_duplicates('text').drop_duplicates('events')
    test_df = pd.DataFrame(ace_test).drop_duplicates('text').drop_duplicates('events')

    train_events: List[EventSentence] = list(convert_events(train_df.to_dict('records')))
    dev_events: List[EventSentence] = list(convert_events(dev_df.to_dict('records')))
    test_events: List[EventSentence] = list(convert_events(test_df.to_dict('records')))

    return train_events, dev_events, test_events


def seed_everything(random_seed: int):
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)


def get_token_indices(span, offsets):
    start_char, end_char = map(int, span.split(":"))
    start_idx, end_idx = None, None

    for i, (start, end) in enumerate(offsets):
        if start <= start_char < end:  # Find first token in span
            start_idx = i
        if start < end_char <= end:  # Find last token in span
            end_idx = i
            break

    return start_idx, end_idx


def encode_dataset(dataset, tokenizer, label2id, modality_p):
    encoded = []

    for sample in dataset:
        toks = tokenizer(sample['text'], truncation=True, return_tensors="pt", add_special_tokens=False, return_offsets_mapping=True, padding="max_length", max_length=254)
        offset_mapping = toks.pop("offset_mapping")

        ner = ['O' for _ in range(0, len(toks['input_ids'][0]))]

        for i in range(0, len(sample['events'])):
            event = sample['events'][i]
            trigger_span = event['trigger'][1][0]

            trigger_start, trigger_end = get_token_indices(trigger_span, offset_mapping[0])
            if modality_p:
                modality = event['event_modality']
                ner[trigger_start] = "B-" + modality
                for j in range(trigger_start + 1, trigger_end + 1):
                    ner[j] = "I-" + modality
            else:
                ner[trigger_start] = "B-trigger"
                for j in range(trigger_start + 1, trigger_end + 1):
                    ner[j] = "I-trigger"

        toks['labels'] = torch.tensor([label2id[label] for label in ner])
        toks["input_ids"] = toks["input_ids"].squeeze(0)
        toks["attention_mask"] = toks["attention_mask"].squeeze(0)
        encoded.append(toks)
    return encoded


def extract_triggers(tokens, labels, tokenizer):
    events = []
    current_trigger = []
    current_label = None

    for token, label in zip(tokens, labels):
        if label == 'O':
            if current_trigger:
                trigger_word = tokenizer.convert_tokens_to_string(current_trigger).strip()
                events.append((current_label, trigger_word))
                current_trigger = []
                current_label = None
            continue

        if label.startswith("B-"):
            if current_trigger:
                trigger_word = tokenizer.convert_tokens_to_string(current_trigger).strip()
                events.append((current_label, trigger_word))
            current_label = label[2:]
            current_trigger = [token]

        elif label.startswith("I-") and current_label == label[2:]:
            current_trigger.append(token)
        else:
            if current_trigger:
                trigger_word = tokenizer.convert_tokens_to_string(current_trigger).strip()
                events.append((current_label, trigger_word))
            current_label = None
            current_trigger = []

    if current_trigger:
        trigger_word = tokenizer.convert_tokens_to_string(current_trigger).strip()
        events.append((current_label, trigger_word))

    return events


def concat_trigger(data: Dataset):
    data["features"] = "[" + data["trigger"] + "]" + ": " + data["text"]
    return data


def calculate_class_weights(labels):
    class_counts = torch.bincount(labels)
    total = labels.size(0)
    weights = total / (len(class_counts) * class_counts)
    return weights


def calc_split(split, model_fact, data_collator, batch_size, device):
    model_fact.to(device)
    model_fact.eval()
    subset = split
    subset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    dataloader = DataLoader(subset, batch_size=batch_size, collate_fn=data_collator)
    all_logits = []
    all_labels = []

    with torch.no_grad():
        i = 0
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model_fact(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
            logits = outputs.logits
            all_logits.append(logits.cpu())
            all_labels.append(batch["labels"].cpu())
            i += 1

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    return all_logits, all_labels
