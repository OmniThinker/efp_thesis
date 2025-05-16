import sys
import os
sys.path.append(os.path.join(os.path.abspath(os.getcwd()), ".."))
import evaluate
from wandb import Table
import wandb
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from transformers import AutoTokenizer, AutoModel
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader as GDataLoader
from torch.utils.data import DataLoader

from torch.optim.lr_scheduler import ReduceLROnPlateau

from certainty import EventSentence, EventType, load_events, id2label, label2id, load_file, CACHE_DIR, GNNCertaintyPredictionModel, seed_everything, RANDOM_SEED, GNN, GNNCombined
from sklearn.metrics import confusion_matrix, classification_report


accuracy = evaluate.load("accuracy")
precision = evaluate.load("precision")
recall = evaluate.load("recall")
f1 = evaluate.load("f1")


class EarlyStopper:
    """
    Credit: https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
    """

    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def extract_unique_roles(events):
    unique_roles = set()

    for event in events:
        for argument in event["arguments"]:
            role = argument[2]
            unique_roles.add(role)

    return sorted(unique_roles)


def get_full_text_embedding(text, tokenizer, model, device):
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True, return_offsets_mapping=True, add_special_tokens=False)
    offsets = tokens.pop("offset_mapping")
    tokens = tokens.to(device)
    with torch.no_grad():
        text_embedding = model(**tokens).last_hidden_state.squeeze(0)
    return (text_embedding.cpu(), offsets)


def get_token_indices(span, offsets):
    start_char, end_char = map(int, span.split(":"))
    start_idx, end_idx = None, None

    for i, (start, end) in enumerate(offsets):
        if start <= start_char < end:
            start_idx = i
        if start < end_char <= end:
            end_idx = i
            break

    return start_idx, end_idx


def event_to_graph(event, tokenizer, model, role_to_idx, role_embedding_layer, device):
    text = event['text']

    text_embedding, offsets = get_full_text_embedding(text, tokenizer, model, device)

    num_tokens, hidden_dim = text_embedding.shape

    trigger_start, trigger_end = get_token_indices(event['trigger_idx'], offsets[0])

    edges = []
    edge_roles = []

    for i in range(num_tokens - 1):
        edges.append((i, i + 1))
        edges.append((i + 1, i))
        edge_roles.append(role_to_idx["Adjacency"])
        edge_roles.append(role_to_idx["Adjacency"])

    for arg_text, arg_position, arg_role in event['arguments']:
        arg_start, arg_end = get_token_indices(arg_position[0], offsets[0])

        if arg_role in role_to_idx:
            role_idx = role_to_idx[arg_role]
        else:
            print(f"Warning: Role '{arg_role}' not in predefined roles.")
            role_idx = role_to_idx["Adjacency"]

        for i in range(arg_start, arg_end + 1):
            for j in range(trigger_start, trigger_end + 1):
                edges.append((j, i))
                edges.append((i, j))
                edge_roles.append(role_idx)
                edge_roles.append(role_idx)

    edge_index = torch.tensor(edges, dtype=torch.long).t()
    edge_roles = torch.tensor(edge_roles, dtype=torch.long)

    edge_attr = role_embedding_layer(edge_roles)

    text = "[" + event["type"] + ", " + event["trigger"] + "]" + ": " + event["text"]
    graph = Data(x=text_embedding, edge_index=edge_index, edge_attr=edge_attr, y=torch.tensor([event['label']]),
                 text=text)
    return graph


def evaluate_model(model, dev_loader, device, criterion):
    all_preds = []
    all_labels = []
    total_loss = 0

    model.eval()
    with torch.no_grad():
        for batch in dev_loader:
            batch.x = batch.x.to(device)
            batch.edge_index = batch.edge_index.to(device)
            batch.edge_attr = batch.edge_attr.to(device)
            batch.y = batch.y.to(device)
            batch.batch = batch.batch.to(device)

            logits = model(batch)

            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
            loss = criterion(logits, batch.y)
            total_loss += loss

    all_preds = list(all_preds)
    all_labels = list(all_labels)

    accuracy_score = accuracy.compute(predictions=all_preds, references=all_labels)
    precision_score = precision.compute(predictions=all_preds, references=all_labels, average=None)
    recall_score = recall.compute(predictions=all_preds, references=all_labels, average=None)
    f1_score = f1.compute(predictions=all_preds, references=all_labels, average=None)

    print(f"Validation loss: {total_loss:.4f}")
    print(f"Accuracy: {accuracy_score['accuracy']:.4f}")
    print(f"Precision per class: {precision_score['precision']}")
    print(f"Recall per class: {recall_score['recall']}")
    print(f"F1-score per class: {f1_score['f1']}")
    wandb.log({
        "eval/accuracy": accuracy_score['accuracy'],
        "eval/other_precision": precision_score["precision"][0],
        "eval/asserted_precision": precision_score["precision"][1],
        "eval/other_recall": recall_score["recall"][0],
        "eval/asserted_recall": recall_score["recall"][1],
        "eval/other_f1": f1_score["f1"][0],
        "eval/asserted_f1": f1_score["f1"][1],
        "eval/loss": total_loss
    })

    return total_loss, f1_score['f1'][0]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--random", type=int, default=RANDOM_SEED)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_bert", type=float, default=5e-6)
    parser.add_argument("--lr_patience", type=int, default=20)
    parser.add_argument("--stopping_patience", type=int, default=10)
    parser.add_argument("--stopping_min_delta", type=int, default=10)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--model", type=str, default='distilbert/distilbert-base-uncased')
    parser.add_argument("--name", type=str, default='GNNSimple')

    args = parser.parse_args()
    return args


def custom_collate(batch):
    return batch


def calculate_class_weights(labels):
    class_counts = torch.bincount(labels)
    total_samples = labels.size(0)
    num_classes = len(class_counts)

    class_counts = class_counts.float()
    class_counts[class_counts == 0] = 1
    weights = total_samples / (num_classes * class_counts)
    weights = weights / weights.sum()

    return weights.to(labels.device)


def main(args):
    random = args.random
    epochs = args.epochs
    lr = args.lr
    lr_bert = args.lr_bert
    lr_patience = args.lr_patience
    stopping_patience = args.stopping_patience
    stopping_min_delta = args.stopping_min_delta
    dropout = args.dropout
    batch_size = args.batch_size
    hidden_dim = args.hidden_dim
    model_name = args.model
    name = args.name

    wandb.init(
        project="certainty",
        name=name,
        config={
            "lr": lr,
            "lr_bert": lr_bert,
            "epochs": epochs,
            "random_state": random,
            "encoder_model": model_name,
            "lr_patience": lr_patience,
            "stopping_patience": stopping_patience,
            "stopping_min_delta": stopping_min_delta,
            "dropout": dropout,
            "batch_size": batch_size,
            "hidden_dim": hidden_dim,
            "decay": "ReduceLROnPlateau",
        },
        dir="..",
    )

    seed_everything(random)
    train_events, dev_events, test_events = load_events('en_train.json', 'en_dev.json', 'en_test.json')
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR, local_files_only=True, trust_remote_code=True)

    tokenizer_model = AutoModel.from_pretrained(
        model_name, cache_dir=CACHE_DIR, local_files_only=True, num_labels=2, id2label=id2label, label2id=label2id, trust_remote_code=True
    )

    ARGUMENT_ROLES = extract_unique_roles(train_events + dev_events + test_events)
    ARGUMENT_ROLES.append("Adjacency")
    role_to_idx = {role: i for i, role in enumerate(ARGUMENT_ROLES)}
    role_embedding_layer = nn.Embedding(len(ARGUMENT_ROLES), 32)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    class_weights = calculate_class_weights(torch.tensor([event['label'] for event in train_events]))
    print(f"Class weights: {class_weights}")

    class_weights = class_weights.to(device)

    tokenizer_model = tokenizer_model.to(device)

    output_dim = tokenizer_model.config.hidden_size

    gnn = GNN(input_dim=output_dim,
              hidden_dim=hidden_dim,
              output_dim=output_dim,
              dropout=dropout).to(device)
    model = GNNCombined(gnn, tokenizer_model, tokenizer, hidden_dim, 2)
    model = model.to(device)

    optimizer = optim.Adam([
        {'params': model.bert.parameters(), 'lr': lr_bert},
        {'params': model.gnn.parameters(), 'lr': lr},
        {'params': list(model.attention.parameters()) + list(model.fc_fusion.parameters()), 'lr': lr},
    ])

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scheduler = ReduceLROnPlateau(optimizer, patience=lr_patience)

    early_stopper = EarlyStopper(patience=stopping_patience, min_delta=stopping_min_delta)

    if os.path.exists('../data/train_graphs.pt'):
        train_graphs = torch.load('../data/train_graphs.pt')
    else:
        train_graphs = [event_to_graph(event, tokenizer, tokenizer_model, role_to_idx, role_embedding_layer, device) for event in train_events]
        torch.save(train_graphs, '../data/train_graphs.pt')

    if os.path.exists('../data/dev_graphs.pt'):
        dev_graphs = torch.load('../data/dev_graphs.pt')
    else:
        dev_graphs = [event_to_graph(event, tokenizer, tokenizer_model, role_to_idx, role_embedding_layer, device) for event in dev_events]
        torch.save(dev_graphs, '../data/dev_graphs.pt')

    if os.path.exists('../data/test_graphs.pt'):
        test_graphs = torch.load('../data/test_graphs.pt')
    else:
        test_graphs = [event_to_graph(event, tokenizer, tokenizer_model, role_to_idx, role_embedding_layer, device) for event in test_events]
        torch.save(test_graphs, '../data/test_graphs.pt')

    train_loader = GDataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    dev_loader = GDataLoader(dev_graphs, batch_size=batch_size, shuffle=False)
    test_loader = GDataLoader(test_graphs, batch_size=batch_size, shuffle=False)

    for epoch in range(epochs):
        print("Training...")
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            batch.x = batch.x.to(device)
            batch.edge_index = batch.edge_index.to(device)
            batch.edge_attr = batch.edge_attr.to(device)
            batch.y = batch.y.to(device)
            batch.batch = batch.batch.to(device)

            logits = model(batch)

            loss = criterion(logits, batch.y)
            loss.backward(retain_graph=True)
            optimizer.step()
            total_loss += loss.item()

        print("Evaluating...")
        wandb.log({"train/learning_rate_gnn": optimizer.param_groups[1]['lr']})
        wandb.log({"train/learning_rate_bert": optimizer.param_groups[0]['lr']})
        wandb.log({"train/learning_rate": optimizer.param_groups[2]['lr']})
        wandb.log({"train/loss": total_loss})
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
        valid_loss, f1_0 = evaluate_model(model, test_loader, device, criterion)

        scheduler.step(f1_0)

        if early_stopper.early_stop(valid_loss):
            break

    evaluate_model(model, tokenizer, id2label, test_loader, test_events, device)


if __name__ == "__main__":
    args = parse_args()
    main(args)
