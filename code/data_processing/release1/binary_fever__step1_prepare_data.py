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

PROMPT = "Is that correct?"


def select_an_incorrect_choice(all_answers, correct_answer):
    remaining_choices = [answer for answer in all_answers if answer != correct_answer]
    assert len(remaining_choices) == len(all_answers) - 1
    wrong_answer = random.choice(remaining_choices)
    assert wrong_answer != correct_answer
    return wrong_answer


def format_hl_fever_datasets(json_dict):
    claim = json_dict["claim"]
    label = json_dict["label"]
    false_option = f"That is not correct. The statement \"{claim}\" is false."
    true_option = f"That is correct. The statement \"{claim}\" is true."
    if label == "refuted":
        formatted_true_answer = false_option
    elif label == "supported":
        formatted_true_answer = true_option
    else:
        assert False
    formatted_answer_choices = [false_option, true_option]
    return claim, formatted_true_answer, formatted_answer_choices


def preprocess_fever(split_name):
    prefix_label = f"hl_fever_{split_name}_"
    ds = load_dataset("pminervini/hl-fever", "v1.0")[split_name]
    json_list = []
    for instance_index in range(len(ds)):
        instance = ds[instance_index]
        claim, formatted_true_answer, available_answers = format_hl_fever_datasets(instance)
        document_id = prefix_label + f"_qid_{instance['id']}" + f"_rowid_{instance_index}"
        if random.randint(0, 1) == 1:
            question_from_conversation = f"{claim} {PROMPT}"
        else:
            question_from_conversation = f"{claim}"
        wrong_answer = select_an_incorrect_choice(available_answers, formatted_true_answer)
        new_dict = {}
        new_dict["question_from_conversation"] = question_from_conversation
        new_dict["provided_solution"] = formatted_true_answer
        new_dict["incorrect_solution"] = wrong_answer
        new_dict["original_line_id"] = document_id
        new_dict["original_claim"] = claim
        json_list.append(new_dict)
    return json_list


def main():
    parser = argparse.ArgumentParser(description="-----[Preprocess FEVER data]-----")
    parser.add_argument("--seed_value", default=0, type=int, help="seed_value")
    parser.add_argument("--output_dir", default="",
                        help="")
    options = parser.parse_args()

    random.seed(options.seed_value)

    for split_name in ["train", "dev"]:
        json_list = preprocess_fever(split_name)
        random.shuffle(json_list)
        output_file = os.path.join(options.output_dir, f"hl_fever_{split_name}_shuffled_as_binary.jsonl")
        data_utils.save_json_lines(output_file, json_list)


if __name__ == "__main__":
    main()
