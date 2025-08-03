# Copyright Reexpress AI, Inc. All rights reserved.

import json
import os
import numpy as np

import argparse
import time
from pathlib import Path
import codecs

import torch
from collections import namedtuple
from pydantic import BaseModel
import asyncio

import data_utils
from datasets import load_dataset
import random

PROMPT = "Please answer the following multiple choice question. Respond only with a single letter choice."


def select_an_incorrect_choice(all_answers, correct_answer):
    remaining_choices = [answer for answer in all_answers if answer != correct_answer]
    assert len(remaining_choices) == len(all_answers) - 1
    wrong_answer = random.choice(remaining_choices)
    assert wrong_answer != correct_answer
    return wrong_answer


def format_mmlu_hf_datasets(json_dict):
    question = json_dict["question"]
    formatted_choice_texts = [f"QUESTION: {question}\n\n"]
    formatted_answer_choices = ["A", "B", "C", "D"]
    choice_i = 0
    formatted_true_answer = ""
    assert len(json_dict["choices"]) == len(formatted_answer_choices), json_dict["choices"]
    assert json_dict["answer"] in range(len(formatted_answer_choices))
    for formatted_label, choice_text in zip(formatted_answer_choices, json_dict["choices"]):
        formatted_choice_texts.append(f"{formatted_label}) {choice_text}\n")
        if choice_i == json_dict["answer"]:
            formatted_true_answer = formatted_label
        choice_i += 1
    assert formatted_true_answer != ""
    return "".join(formatted_choice_texts), formatted_true_answer, formatted_answer_choices


def format_mmlu_pro_hf_datasets(json_dict):
    question = json_dict["question"]
    formatted_choice_texts = [f"QUESTION: {question}\n\n"]
    formatted_answer_choices = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    choice_i = 0
    formatted_true_answer = ""
    assert len(json_dict["options"]) <= len(formatted_answer_choices), \
        f'{len(json_dict["options"])}, {len(formatted_answer_choices)}, {json_dict}'
    if len(json_dict["options"]) < len(formatted_answer_choices):
        print(f"WARNING: option set only contains {len(json_dict['options'])} items: {json_dict}")
    assert json_dict["answer_index"] in range(len(formatted_answer_choices))
    for formatted_label, choice_text in zip(formatted_answer_choices, json_dict["options"]):
        formatted_choice_texts.append(f"{formatted_label}) {choice_text}\n")
        if choice_i == json_dict["answer_index"]:
            formatted_true_answer = formatted_label
            if formatted_true_answer != json_dict["answer"]:
                print(f"WARNING: The index field appears to be mismatched. Reverting to the answer letter. {json_dict}")
                formatted_true_answer = json_dict["answer"]
        choice_i += 1
    assert formatted_true_answer != ""
    return "".join(formatted_choice_texts), formatted_true_answer, formatted_answer_choices


def preprocess_mmlu_pro(split_name):
    prefix_label = f"mmlu_pro_{split_name}_"
    ds = load_dataset("TIGER-Lab/MMLU-Pro")[split_name]
    json_list = []
    for instance_index in range(len(ds)):
        instance = ds[instance_index]
        formatted_choice_texts, formatted_true_answer, available_answers = format_mmlu_pro_hf_datasets(instance)
        document_id = prefix_label + instance["category"] + f"_qid_{instance['question_id']}" + f"_rowid_{instance_index}"
        question_from_conversation = f"{PROMPT} {formatted_choice_texts}"
        wrong_answer = select_an_incorrect_choice(available_answers, formatted_true_answer)
        new_dict = {}
        new_dict["question_from_conversation"] = question_from_conversation
        new_dict["provided_solution"] = formatted_true_answer
        new_dict["incorrect_solution"] = wrong_answer
        new_dict["original_line_id"] = document_id
        json_list.append(new_dict)
    return json_list


def preprocess_mmlu(split_name):
    prefix_label = f"mmlu_{split_name}_"
    ds = load_dataset("cais/mmlu", "all")[split_name]
    json_list = []
    for instance_index in range(len(ds)):
        instance = ds[instance_index]
        formatted_choice_texts, formatted_true_answer, available_answers = format_mmlu_hf_datasets(instance)
        document_id = prefix_label + instance["subject"] + f"_rowid_{instance_index}"
        question_from_conversation = f"{PROMPT} {formatted_choice_texts}"
        wrong_answer = select_an_incorrect_choice(available_answers, formatted_true_answer)
        new_dict = {}
        new_dict["question_from_conversation"] = question_from_conversation
        new_dict["provided_solution"] = formatted_true_answer
        new_dict["incorrect_solution"] = wrong_answer
        new_dict["original_line_id"] = document_id
        json_list.append(new_dict)
    return json_list


def main():
    parser = argparse.ArgumentParser(description="-----[Preprocess multiple-choice data]-----")
    parser.add_argument("--seed_value", default=0, type=int, help="seed_value")
    parser.add_argument("--output_dir", default="",
                        help="")
    options = parser.parse_args()

    random.seed(options.seed_value)

    for split_name in ["validation", "test"]:
        json_list = preprocess_mmlu_pro(split_name)
        random.shuffle(json_list)
        output_file = os.path.join(options.output_dir, f"mmlu_pro_{split_name}_shuffled_as_binary.jsonl")
        data_utils.save_json_lines(output_file, json_list)

    for split_name in ['test', 'validation', 'dev', 'auxiliary_train']:
        json_list = preprocess_mmlu(split_name)
        random.shuffle(json_list)
        output_file = os.path.join(options.output_dir, f"mmlu_{split_name}_shuffled_as_binary.jsonl")
        data_utils.save_json_lines(output_file, json_list)


if __name__ == "__main__":
    main()
