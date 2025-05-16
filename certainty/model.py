import torch.nn as nn
import torch
from torch_geometric.nn import GATConv, GATv2Conv, BatchNorm, global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.data import Data, Batch

import torch.nn.functional as F
from transformers import PreTrainedModel, AutoModel
from transformers.modeling_outputs import TokenClassifierOutput
from .types import EventType
from typing import get_args


class GNNCertaintyPredictionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, bert_model, bert_tokenizer,
                 role_to_idx,
                 role_embedding_layer,
                 dropout=0.5):
        super(GNNCertaintyPredictionModel, self).__init__()
        self.role_to_idx = role_to_idx
        self.role_embedding_layer = role_embedding_layer
        self.bert_model = bert_model
        self.bert_tokenizer = bert_tokenizer

        self.gat1 = GATConv(input_dim, hidden_dim, heads=2, concat=True)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 4, output_dim)

    def forward(self, events, device):
        batch_graphs = []

        for event in events:
            graph = self.event_to_graph(event, device)
            batch_graphs.append(graph)

        batch_graph = Batch.from_data_list(batch_graphs)

        batch_graph.x = batch_graph.x.to(device)
        batch_graph.edge_index = batch_graph.edge_index.to(device)
        batch_graph.edge_attr = batch_graph.edge_attr.to(device)
        batch_graph.batch = batch_graph.batch.to(device)

        x = self.gat1(batch_graph.x, batch_graph.edge_index, batch_graph.edge_attr)
        x = F.relu(x)
        x = self.dropout(x)
        x = global_mean_pool(x, batch_graph.batch)

        logits = self.fc(x)
        return logits

    def event_to_graph(self, event, device):
        text = event['text']

        text_embedding, offsets = self.get_full_text_embedding(text, device)
        num_tokens, hidden_dim = text_embedding.shape

        trigger_start, trigger_end = self.get_token_indices(event['trigger_idx'], offsets[0])

        edges = []
        edge_roles = []

        for i in range(num_tokens - 1):
            edges.append((i, i + 1))
            edges.append((i + 1, i))
            edge_roles.append(self.role_to_idx["Adjacency"])
            edge_roles.append(self.role_to_idx["Adjacency"])

        for arg_text, arg_position, arg_role in event['arguments']:
            arg_start, arg_end = self.get_token_indices(arg_position[0], offsets[0])

            if arg_role in self.role_to_idx:
                role_idx = self.role_to_idx[arg_role]
            else:
                print(f"Warning: Role '{arg_role}' not in predefined roles.")
                role_idx = self.role_to_idx["Adjacency"]

            for i in range(arg_start, arg_end + 1):
                for j in range(trigger_start, trigger_end + 1):
                    edges.append((j, i))
                    edges.append((i, j))
                    edge_roles.append(role_idx)
                    edge_roles.append(role_idx)

        edge_index = torch.tensor(edges, dtype=torch.long).t().to(device)
        edge_roles = torch.tensor(edge_roles, dtype=torch.long).to(device)

        edge_attr = self.role_embedding_layer(edge_roles)

        graph = Data(x=text_embedding, edge_index=edge_index, edge_attr=edge_attr, y=torch.tensor([event['label']]),
                     text=text)
        return graph

    def get_full_text_embedding(self, text, device):
        tokens = self.bert_tokenizer(text, return_tensors="pt", padding=True, truncation=True, return_offsets_mapping=True, add_special_tokens=False)
        tokens = tokens.to(device)
        offsets = tokens.pop("offset_mapping")
        text_embedding = self.bert_model(**tokens).last_hidden_state.squeeze(0)
        return (text_embedding, offsets)

    def get_token_indices(self, span, offsets):
        start_char, end_char = map(int, span.split(":"))
        start_idx, end_idx = None, None

        for i, (start, end) in enumerate(offsets):
            if start <= start_char < end:
                start_idx = i
            if start < end_char <= end:
                end_idx = i
                break

        return start_idx, end_idx


class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        super(GNN, self).__init__()

        self.gat1 = GATv2Conv(input_dim, hidden_dim, heads=2, concat=True, edge_dim=32)
        self.proj_gnn = nn.Linear(hidden_dim * 2, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, batch):
        x = self.gat1(batch.x, batch.edge_index, batch.edge_attr)
        x = F.relu(x)
        x = self.dropout(x)

        x = global_mean_pool(x, batch.batch)

        logits = self.proj_gnn(x)

        return logits


class GNNCombined(nn.Module):
    def __init__(self, gnn_model, bert_model, tokenizer, hidden_dim, output_dim, dropout=0.5):
        super(GNNCombined, self).__init__()
        self.bert = bert_model
        self.gnn = gnn_model
        self.tokenizer = tokenizer
        bert_hidden_dim = self.bert.config.hidden_size

        combined_dim = bert_hidden_dim + bert_hidden_dim

        self.attention = nn.MultiheadAttention(embed_dim=combined_dim, num_heads=2, batch_first=True)

        self.fc_fusion = nn.Linear(combined_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, batch):
        gnn_out = self.gnn(batch)

        encoded_text = self.tokenizer(batch.text, padding=True, truncation=True, return_tensors="pt")
        encoded_text = {k: v.to(next(self.bert.parameters()).device) for k, v in encoded_text.items()}
        bert_out = self.bert(**encoded_text).last_hidden_state[:, 0, :]

        combined = torch.cat([gnn_out, bert_out], dim=1).unsqueeze(1)
        combined = self.dropout(combined)

        att, _ = self.attention(combined, combined, combined)
        att = att.squeeze(1)

        logits = self.fc_fusion(att)

        return logits


class HierarchicalFactualityModel(PreTrainedModel):
    def __init__(self, config, model_name):
        super().__init__(config)
        self.num_labels_factuality = 5
        self.num_labels_event_type = len(get_args(EventType)) + 1

        self.transformer = AutoModel.from_pretrained(model_name, config=config)

        for param in self.transformer.parameters():
            param.requires_grad = False

        hidden_size = config.hidden_size

        self.factuality_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, self.num_labels_factuality)
        )

        self.event_type_classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.2),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.1),
            nn.GELU(),
            nn.Linear(hidden_size, self.num_labels_event_type)
        )

        self.init_weights()

    def unfreeze_weights(self):
        for param in self.transformer.parameters():
            param.requires_grad = True
