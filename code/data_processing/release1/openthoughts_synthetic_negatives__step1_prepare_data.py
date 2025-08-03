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


def construct_data():
    ds = load_dataset("open-thoughts/OpenThoughts2-1M")
    json_list = []
    json_list_streamlined = []
    count_malformed = 0
    for instance_index in range(len(ds['train'])):
        instance = ds["train"][instance_index]
        assert len(instance["conversations"]) == 2
        raw_question = instance["question"]
        question_from_conversation = instance["conversations"][0]["value"]
        # if raw_question != question_from_conversation:
        #     print(f"WARNING: Question mismatch: {raw_question} || {question_from_conversation}")
        if instance["conversations"][1]["value"].count('</think>') != 1:
            count_malformed += 1
            continue
        final_think_tag_index = instance["conversations"][1]["value"].find('</think>') + len('</think>')
        provided_solution = instance["conversations"][1]["value"][final_think_tag_index:].strip()
        instance["question_from_conversation"] = question_from_conversation
        instance["provided_solution"] = provided_solution
        instance["original_line_id"] = instance_index
        json_list.append(instance)
        new_dict = {}
        new_dict["question_from_conversation"] = question_from_conversation
        new_dict["provided_solution"] = provided_solution
        new_dict["original_line_id"] = instance_index
        json_list_streamlined.append(new_dict)
    print(f"Total unexpected formatting so skipped: {count_malformed}")
    return json_list, json_list_streamlined


def main():
    parser = argparse.ArgumentParser(description="-----[Preprocess OpenThoughts2-1M]-----")
    parser.add_argument("--seed_value", default=0, type=int, help="seed_value")
    parser.add_argument("--output_file", default="",
                        help="")
    parser.add_argument("--output_file_streamlined", default="",
                        help="")
    options = parser.parse_args()

    random.seed(options.seed_value)

    json_list, json_list_streamlined = construct_data()
    random.shuffle(json_list_streamlined)
    data_utils.save_json_lines(options.output_file, json_list)
    data_utils.save_json_lines(options.output_file_streamlined, json_list_streamlined)


if __name__ == "__main__":
    main()
