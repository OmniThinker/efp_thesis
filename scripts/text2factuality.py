import sys
import os
sys.path.append(os.path.join(os.path.abspath(os.getcwd()), ".."))
import pandas as pd
import numpy as np
import re
import wandb
import argparse

import torch
from transformers import (
    Trainer,
    Seq2SeqTrainingArguments,
    PreTrainedTokenizer,
    EvalPrediction,
    Seq2SeqTrainer,
    default_data_collator,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    MBartTokenizer,
    default_data_collator,
    AutoConfig,
    EarlyStoppingCallback
)
from datasets import Dataset
from certainty import load_file, CACHE_DIR, seed_everything, RANDOM_SEED, EventType, TRAIN_FILENAME, TEST_FILENAME, DEV_FILENAME
OUTPUT_DIR = "../models/text2factuality"

from dataclasses import dataclass, field

from transformers.trainer_utils import get_last_checkpoint, is_main_process
from typing import Union, List, Dict, Tuple, Any, Optional
import torch.nn as nn


class FactualitySchema:
    def __init__(self, type_list):
        self.type_list = type_list
        self.role_list = []
        self.type_role_dict = {}


@dataclass
class SumLabelSmoother:
    """
    Adds label-smoothing on a pre-computed output from a Transformers model.

    Args:
        epsilon (:obj:`float`, `optional`, defaults to 0.1):
            The label smoothing factor.
        ignore_index (:obj:`int`, `optional`, defaults to -100):
            The index in the labels to ignore when computing the loss.
    """

    epsilon: float = 0.1
    ignore_index: int = -100
    factuality_labels: Optional[Any] = None
    factuality_weight: float = 3.08
    tokenizer: Any = None
    check_mask = True

    def build_factuality_mask(self, labels):
        labels = labels.cpu().numpy()
        mask = np.zeros_like(labels, dtype=bool)
        for label_seq in self.factuality_labels.values():
            L = len(label_seq)
            for b in range(labels.shape[0]):
                for i in range(labels.shape[1] - L + 1):
                    if (labels[b, i:i + L] == label_seq).all():
                        mask[b, i:i + L] = True

        return torch.tensor(mask, device="cuda", dtype=torch.bool)

    def __call__(self, model_output, labels):
        logits = model_output["logits"] if isinstance(model_output, dict) else model_output[0]
        log_probs = -torch.nn.functional.log_softmax(logits, dim=-1)

        if labels.dim() == log_probs.dim() - 1:
            labels = labels.unsqueeze(-1)

        padding_mask = labels.eq(self.ignore_index)
        labels = labels.clamp(min=0)

        factuality_mask = self.build_factuality_mask(labels.squeeze(-1))

        if self.check_mask:
            print(factuality_mask)
        nll_loss = log_probs.gather(dim=-1, index=labels)
        smoothed_loss = log_probs.sum(dim=-1, keepdim=True)
        if factuality_mask.sum() == 0 and self.check_mask:
            print("No factuality tokens found in label sequence!")
            self.check_mask = False
        elif self.check_mask:
            print(f"Found {factuality_mask.sum().item()} factuality tokens in current batch.")
            self.check_mask = False

        nll_loss.masked_fill_(padding_mask, 0.0)
        smoothed_loss.masked_fill_(padding_mask, 0.0)

        factuality_weight_mask = factuality_mask.float() * (self.factuality_weight - 1.0) + 1.0

        factuality_weight_mask = factuality_weight_mask.unsqueeze(-1)
        nll_loss *= factuality_weight_mask

        smoothed_loss *= factuality_weight_mask

        nll_loss = nll_loss.sum()
        smoothed_loss = smoothed_loss.sum()

        total_tokens = (~padding_mask).sum().clamp(min=1)

        eps_i = self.epsilon / log_probs.size(-1)
        loss = ((1.0 - self.epsilon) * nll_loss + eps_i * smoothed_loss) / total_tokens
        return loss


def get_label_name_tree(label_name_list, tokenizer, end_symbol='<end>'):
    sub_token_tree = dict()

    label_tree = dict()
    for typename in label_name_list:
        after_tokenized = tokenizer.encode(typename, add_special_tokens=False)

        print(f"[DEBUG] Label: {typename}")
        print(f"[DEBUG] Tokenized: {tokenizer.tokenize(typename)}")
        print(f"[DEBUG] Token IDs: {after_tokenized}")
        label_tree[typename] = after_tokenized

    for _, sub_label_seq in label_tree.items():
        parent = sub_token_tree
        for value in sub_label_seq:
            if value not in parent:
                parent[value] = dict()
            parent = parent[value]

        parent[end_symbol] = None

    return sub_token_tree


class PrefixTree:
    def __init__(self, label_name_list, tokenizer, end_symbol='<end>'):
        self.label_name_list = label_name_list
        self._tokenizer = tokenizer
        self.label_name_tree = get_label_name_tree(label_name_list, tokenizer, end_symbol)
        self._end_symbol = end_symbol

    def is_end_of_tree(self, tree: Dict):
        return len(tree) == 1 and self._end_symbol in tree


def match_sublist(the_list, to_match):
    """

    :param the_list: [1, 2, 3, 4, 5, 6, 1, 2, 4, 5]
    :param to_match: [1, 2]
    :return:
        [(0, 1), (6, 7)]
    """
    len_to_match = len(to_match)
    matched_list = list()
    for index in range(len(the_list) - len_to_match + 1):
        if to_match == the_list[index:index + len_to_match]:
            matched_list += [(index, index + len_to_match - 1)]
    return matched_list


def find_bracket_position(generated_text, _type_start, _type_end):
    bracket_position = {_type_start: list(), _type_end: list()}
    for index, char in enumerate(generated_text):
        if char in bracket_position:
            bracket_position[char] += [index]
    return bracket_position


def generated_search_src_sequence(generated, src_sequence, end_sequence_search_tokens=None):

    if len(generated) == 0:
        return src_sequence

    matched_tuples = match_sublist(the_list=src_sequence, to_match=generated)

    valid_token = list()
    for _, end in matched_tuples:
        next_index = end + 1
        if next_index < len(src_sequence):
            valid_token += [src_sequence[next_index]]

    if end_sequence_search_tokens:
        valid_token += end_sequence_search_tokens

    return valid_token


class ConstraintDecoder:
    def __init__(self, tokenizer, source_prefix):
        self.tokenizer = tokenizer
        self.source_prefix = source_prefix
        self.source_prefix_tokenized = tokenizer.encode(source_prefix,
                                                        add_special_tokens=False) if source_prefix else []

    def get_state_valid_tokens(self, src_sentence: List[str], tgt_generated: List[str]) -> List[str]:
        pass

    def constraint_decoding(self, src_sentence, tgt_generated):
        if self.source_prefix_tokenized:
            # Remove Source Prefix for Generation
            src_sentence = src_sentence[len(self.source_prefix_tokenized):]

        valid_token_ids = self.get_state_valid_tokens(
            src_sentence.tolist(),
            tgt_generated.tolist()
        )

        if not valid_token_ids:
            print(f"WARNING: No valid tokens returned for src={src_sentence}, tgt={tgt_generated}")
            return list(range(self.tokenizer.vocab_size))

        return valid_token_ids


class TreeConstraintDecoder(ConstraintDecoder):
    def __init__(self, tokenizer, type_schema, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        self.tree_end = '<tree-end>'
        self.type_tree = get_label_name_tree(
            type_schema.type_list, self.tokenizer, end_symbol=self.tree_end)

        print(f"[DEBUG] Type tree keys: {list(self.type_tree.keys())}")

        self.role_tree = get_label_name_tree(
            type_schema.role_list, self.tokenizer, end_symbol=self.tree_end)

        self.type_start = self.tokenizer.convert_tokens_to_ids(["("])[0]
        self.type_end = self.tokenizer.convert_tokens_to_ids([")"])[0]

    def check_state(self, tgt_generated):
        if tgt_generated[-1] == self.tokenizer.pad_token_id:
            return 'start', -1

        special_token_set = {self.type_start, self.type_end}
        special_index_token = list(
            filter(lambda x: x[1] in special_token_set, list(enumerate(tgt_generated))))

        last_special_index, last_special_token = special_index_token[-1]

        if len(special_index_token) == 1:
            if last_special_token != self.type_start:
                return 'error', 0

        bracket_position = find_bracket_position(
            tgt_generated, _type_start=self.type_start, _type_end=self.type_end)
        start_number, end_number = len(bracket_position[self.type_start]), len(
            bracket_position[self.type_end])

        if start_number == end_number:
            return 'end_generate', -1
        if start_number == end_number + 1:
            state = 'start_first_generation'
        elif start_number == end_number + 2:
            state = 'generate_trigger'
        elif start_number == end_number + 3:
            state = 'generate_role'
        else:
            state = 'error'
        return state, last_special_index

    def search_prefix_tree_and_sequence(self, generated: List[str], prefix_tree: Dict, src_sentence: List[str],
                                        end_sequence_search_tokens: List[str] = None):
        """
        Generate Type Name + Text Span
        :param generated:
        :param prefix_tree:
        :param src_sentence:
        :param end_sequence_search_tokens:
        :return:
        """
        tree = prefix_tree
        for index, token in enumerate(generated):
            tree = tree[token]
            is_tree_end = len(tree) == 1 and self.tree_end in tree

            if is_tree_end:
                valid_token = generated_search_src_sequence(
                    generated=generated[index + 1:],
                    src_sequence=src_sentence,
                    end_sequence_search_tokens=end_sequence_search_tokens,
                )
                return valid_token

            if self.tree_end in tree:
                try:
                    valid_token = generated_search_src_sequence(
                        generated=generated[index + 1:],
                        src_sequence=src_sentence,
                        end_sequence_search_tokens=end_sequence_search_tokens,
                    )
                    return valid_token
                except IndexError:
                    # Still search tree
                    continue

        valid_token = list(tree.keys())
        return valid_token

    def get_state_valid_tokens(self, src_sentence, tgt_generated):
        """

        :param src_sentence:
        :param tgt_generated:
        :return:
            List[str], valid token list
        """
        if self.tokenizer.eos_token_id in src_sentence:
            src_sentence = src_sentence[:src_sentence.index(
                self.tokenizer.eos_token_id)]

        state, index = self.check_state(tgt_generated)

        if state == 'error':
            print("Error:")
            print("Src:", src_sentence)
            print("Tgt:", tgt_generated)
            valid_tokens = [self.tokenizer.eos_token_id]

        elif state == 'start':
            valid_tokens = [self.type_start]

        elif state == 'start_first_generation':
            valid_tokens = [self.type_start, self.type_end]

        elif state == 'generate_trigger':

            if tgt_generated[-1] == self.type_start:
                # Start Event Label
                return list(self.type_tree.keys())

            elif tgt_generated[-1] == self.type_end:
                # EVENT_TYPE_LEFT: Start a new role
                # EVENT_TYPE_RIGHT: End this event
                return [self.type_start, self.type_end]
            else:
                valid_tokens = self.search_prefix_tree_and_sequence(
                    generated=tgt_generated[index + 1:],
                    prefix_tree=self.type_tree,
                    src_sentence=src_sentence,
                    end_sequence_search_tokens=[self.type_start, self.type_end]
                )

        elif state == 'generate_role':

            if tgt_generated[-1] == self.type_start:
                # Start Role Label
                return list(self.role_tree.keys())

            generated = tgt_generated[index + 1:]
            valid_tokens = self.search_prefix_tree_and_sequence(
                generated=generated,
                prefix_tree=self.role_tree,
                src_sentence=src_sentence,
                end_sequence_search_tokens=[self.type_end]
            )

        elif state == 'end_generate':
            valid_tokens = [self.tokenizer.eos_token_id]

        else:
            raise NotImplementedError(
                'State `%s` for %s is not implemented.' % (state, self.__class__))
        return valid_tokens


def get_constraint_decoder(tokenizer, type_schema, decoding_schema, source_prefix=None):
    return TreeConstraintDecoder(tokenizer=tokenizer, type_schema=type_schema, source_prefix=source_prefix)


@dataclass
class ConstraintSeq2SeqTrainingArguments(Seq2SeqTrainingArguments):
    """
    Parameters:
        constraint_decoding (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to use Constraint Decoding
        structure_weight (:obj:`float`, `optional`, defaults to :obj:`None`):
    """
    constraint_decoding: bool = field(default=False, metadata={"help": "Whether to Constraint Decoding or not."})
    label_smoothing_sum: bool = field(default=False,
                                      metadata={"help": "Whether to use sum token loss for label smoothing"})


class ConstraintSeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(self, decoding_type_schema=None, decoding_format='tree', source_prefix=None, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.decoding_format = decoding_format
        self.decoding_type_schema = decoding_type_schema

        if True:
            factuality_labels = {
                "Asserted": self.tokenizer.encode("Asserted", add_special_tokens=False),
                "Other": self.tokenizer.encode("Other", add_special_tokens=False),
            }
            print("Factuality label token sequences:", factuality_labels)

            self.label_smoother = SumLabelSmoother(
                epsilon=self.args.label_smoothing_factor,
                factuality_labels=factuality_labels,
                factuality_weight=50,  # 3.08
                tokenizer=self.tokenizer
            )
            print('Using SumLabelSmoother')
        else:
            self.label_smoother = None

        print(self.label_smoother)

        if self.args.constraint_decoding:
            self.constraint_decoder = get_constraint_decoder(tokenizer=self.tokenizer,
                                                             type_schema=self.decoding_type_schema,
                                                             decoding_schema=self.decoding_format,
                                                             source_prefix=source_prefix)
            print(self.constraint_decoder)
        else:
            self.constraint_decoder = None
        print("Trainer initialized! Training will use constraint decoding?", self.args.constraint_decoding)

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        def prefix_allowed_tokens_fn(batch_id, sent):
            src_sentence = inputs['input_ids'][batch_id]
            return self.constraint_decoder.constraint_decoding(src_sentence=src_sentence,
                                                               tgt_generated=sent)

        return super().prediction_step(
            model=model,
            inputs=inputs,
            prediction_loss_only=prediction_loss_only,
            ignore_keys=ignore_keys,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn if self.constraint_decoder else None,
        )


def parse_sample(sample):
    results = []
    i = 0
    length = len(sample)

    while i < length:
        try:
            if sample[i] != '(':
                i += 1
                continue
            if i + 1 >= length or sample[i + 1] != '(':
                i += 1
                continue
            i += 2

            while i < length and sample[i].isspace():
                i += 1

            start = i
            while i < length and not sample[i].isspace():
                i += 1
            label = sample[start:i].strip()

            if label not in {"Asserted", "Other"}:
                continue

            while i < length and sample[i].isspace():
                i += 1

            trigger_start = i
            while i < length and sample[i] != ')':
                i += 1
            trigger = sample[trigger_start:i].strip()

            if i >= length or sample[i] != ')':
                continue
            i += 1
            if i >= length or sample[i] != ')':
                continue
            i += 1

            results.append((label, trigger))

        except Exception as e:
            print(f"Failed to parse at index {i}: {str(e)}")
            i += 1
            continue

    return results


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
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--random", type=int, default=RANDOM_SEED)
    parser.add_argument("--model_name", type=str, default='t5-small')
    parser.add_argument("--name", type=str, default='text2factuality')
    parser.add_argument("--batch_size", type=int, default=16)

    args = parser.parse_args()
    return args


def main(args):
    random = args.random
    lr = args.lr
    lr_strategy = args.lr_strategy
    epochs = args.epochs
    text_column = "text"
    summary_column = "factuality"
    max_target_length = 128
    pad_to_max_length = True
    batch_size = args.batch_size
    ignore_pad_token_for_loss = True
    name = args.name
    model_name = args.model_name
    prefix = "predict (factuality trigger) pairs for all events. Example: ((Other hit) (Other going)): "
    seed_everything(random)
    train = load_file('en_train.json')
    # TEST SET
    dev = load_file('en_test.json')
    # TEST SET

    train = pd.DataFrame(train).drop_duplicates('text').drop_duplicates('events').to_dict("records")
    dev = pd.DataFrame(dev).drop_duplicates('text').drop_duplicates('events').to_dict("records")

    new_train = []

    for sample in train:
        pairs = "("

        for i, event in enumerate(sample['events']):
            # if event['event_genericity'] == 'Generic':
            #     pair = "( " + "Other" + " " + event['trigger'][0][0] + ")"
            # else:
            pair = "( " + event['event_modality'] + " " + event['trigger'][0][0] + ")"
            if i > 0:
                pairs += " "

            pairs += str(pair)

        pairs += ")"

        new_train.append({'factuality': pairs,
                          'text': sample['text']})

    new_dev = []
    for sample in dev:
        pairs = "("
        for i, event in enumerate(sample['events']):
            # if event['event_genericity'] == 'Generic':
            #     pair = "( " + "Other" + " " + event['trigger'][0][0] + ")"
            # else:
            pair = "( " + event['event_modality'] + " " + event['trigger'][0][0] + ")"

            if i > 0:
                pairs += " "
            pairs += str(pair)
        pairs += ")"
        new_dev.append({'factuality': pairs,
                        'text': sample['text']})

    train_dataset = Dataset.from_pandas(pd.DataFrame(new_train)).shuffle(seed=random)
    dev_dataset = Dataset.from_pandas(pd.DataFrame(new_dev))

    config = AutoConfig.from_pretrained(
        model_name,
        cache_dir=CACHE_DIR,
        # local_files_only=True,
        trust_remote_code=True
    )
    config.max_length = max_target_length

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=CACHE_DIR,
        use_fast=True
    )

    to_remove_token_list = list()
    if tokenizer.bos_token:
        to_remove_token_list += [tokenizer.bos_token]
    if tokenizer.eos_token:
        to_remove_token_list += [tokenizer.eos_token]
    if tokenizer.pad_token:
        to_remove_token_list += [tokenizer.pad_token]

    print(tokenizer.convert_ids_to_tokens([188]))
    print(tokenizer.decode([188]))

    print(tokenizer.convert_ids_to_tokens([188, 7, 7, 49, 1054]))
    print(tokenizer.decode([188, 7, 7, 49, 1054]))
    print("Token IDs for 'asserted':", tokenizer.encode("asserted", add_special_tokens=False))

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        config=config,
        cache_dir=CACHE_DIR
    )

    if model.config.decoder_start_token_id is None and isinstance(tokenizer, MBartTokenizer):
        model.config.decoder_start_token_id = tokenizer.lang_code_to_id['en']
    if model.config.decoder_start_token_id is None:
        raise ValueError(
            "Make sure that `config.decoder_start_token_id` is correctly defined")

    decoding_type_schema = FactualitySchema(['Asserted', 'Other'])
    padding = "max_length" if pad_to_max_length else False

    def preprocess_function(examples):
        inputs = examples[text_column]
        targets = examples[summary_column]
        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(
            inputs, max_length=1024, padding=padding, truncation=True, return_tensors="pt"
        )

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True, return_tensors="pt")

        if padding == "max_length" and ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=1,
        remove_columns=train_dataset.column_names
    )
    dev_dataset = dev_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=1,
        remove_columns=dev_dataset.column_names
    )

    train_dataset.set_format("pt", columns=["input_ids", "labels", "attention_mask"], output_all_columns=True)
    dev_dataset.set_format("pt", columns=["input_ids", "labels", "attention_mask"], output_all_columns=True)

    label_pad_token_id = - \
        100 if True else tokenizer.pad_token_id

    eval_set = dev

    def compute_metrics(eval_preds):
        print("Inside compute_mterics")
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        print("decoding")
        decoded_preds = tokenizer.batch_decode(
            preds, skip_special_tokens=False)

        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        def clean_str(x_str):
            if not isinstance(x_str, str):
                pass
            for to_remove_token in to_remove_token_list:
                x_str = x_str.replace(to_remove_token, '')
            return x_str.strip()

        print("cleaning strings")
        decoded_preds = [clean_str(x) for x in decoded_preds]

        parsed_pred = list(map(parse_sample, decoded_preds))
        parsed_true = []
        for sample in eval_set[:len(parsed_pred)]:
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
        print("Should save csv")
        m = model_name.split("/")[-1]
        df.to_csv(f"../results/text2factuality/test-{random}-{m}-{lr}-{lr_strategy}.csv")
        print("Returning")
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
            "other_f1": other_f1
        }

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if True else None,
    )

    wandb.init(
        project="certainty",
        name=name,
        config={
            "lr": lr,
            "lr_strategy": lr_strategy,
            "epochs": epochs,
            "random_state": random,
            "model": model_name
        },
        dir="..",
    )

    training_args = ConstraintSeq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=lr,
        lr_scheduler_type=lr_strategy,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=2,
        constraint_decoding=True,
        num_train_epochs=epochs,
        predict_with_generate=True,
        label_smoothing_factor=0.05,
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
        save_strategy="epoch",
        report_to="wandb",
        save_total_limit=1,
        weight_decay=0.01,
        warmup_ratio=0.1
    )

    trainer = ConstraintSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        decoding_type_schema=decoding_type_schema,
        decoding_format='tree',
        source_prefix=prefix,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )

    trainer.train()


if __name__ == "__main__":
    args = parse_args()
    main(args)
