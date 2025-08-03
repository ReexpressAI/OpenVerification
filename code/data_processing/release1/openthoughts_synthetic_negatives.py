# Copyright Reexpress AI, Inc. All rights reserved.

from typing import Any, Callable, List, Tuple
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

# env variables
USE_AZURE_01 = int(os.getenv("USE_AZURE_01", "1"))
if USE_AZURE_01 == 1:
    from openai import AzureOpenAI
    kAPI_VERSION = "2024-12-01-preview"
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=kAPI_VERSION,
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )
    LOG_PROB_MODEL = os.getenv("GPT41_2025_04_14_AZURE_DEPLOYMENT_NAME")
    REASONING_MODEL = os.getenv("O4_MINI_2025_04_16_AZURE_DEPLOYMENT_NAME")
else:
    from openai import OpenAI
    client = OpenAI()
    LOG_PROB_MODEL = "gpt-4.1-2025-04-14"
    REASONING_MODEL = "o4-mini-2025-04-16"

kSLEEP_CONSTANT = 40
VERIFICATION_CLASSIFICATION_KEY = "verification_classification"
CONFIDENCE_IN_CLASSIFICATION_KEY = "confidence_in_classification"
SHORT_EXPLANATION_FOR_CLASSIFICATION_CONFIDENCE_KEY = "short_explanation_for_classification_confidence"

SHORT_EXPLANATION_FOR_CLASSIFICATION_CONFIDENCE__DEFAULT_ERROR = "Unfortunately, I am unable to verify that response. Please consider providing additional clarification and/or additional references, results, or other information that may assist in the verification process."

class IncorrectSolutionGeneratorWithConfidenceAndExplanation(BaseModel):
    correct_solution_is_correct_classification: bool
    confidence_in_classification: float
    short_explanation_for_classification_confidence: str
    incorrect_solution: str
    short_explanation_why_new_incorrect_solution_is_wrong: str

DATA_GENERATION_SYSTEM_MESSAGE_WITH_EXPLANATION = """
You are a helpful assistant that creates educational test problems for students in science and engineering fields. You are given a question and are presented with a correct solution. Your task is to create a distractor solution by introducing a subtle error or mistake in the provided solution. It is important that your incorrect solution is in fact unambiguously wrong, but try not to make the error obvious for the student. The question is contained within the XML tags <question> and </question>, and the correct solution is contained within the XML tags <correct_solution> and </correct_solution>. If you find that the provided solution actually already contains an error, please briefly specify why, but then also provide an additional incorrect solution. Please structure your response using the provided JSON format. If the correct solution is itself wrong, respond with False for the correct_solution_is_correct_classification field, but otherwise respond with True. Provide a confidence estimate in this classification as a probability between 0 and 1, where a probability of 0 indicates no confidence that the correct solution is correct and a probability of 1 indicates 100% confidence in the correctness of the original solution. If the original solution is wrong, briefly explain why using the short_explanation_for_classification_confidence field. Next, provide your new incorrect solution for the incorrect_solution field. Your new incorrect solution should be roughly the same length as the provided correct solution and should follow the general format of the provided correct solution. The text you provide for the incorrect_solution JSON key should not mention why the solution is incorrect, and it should not provide any hints to the student. If the original correct solution is a multiple choice question, your incorrect solution should not choose the same answer letter. Finally, briefly explain why your new incorrect solution is incorrect in the short_explanation_why_new_incorrect_solution_is_wrong field.
"""


def create_negative(query_and_response_string: str): # -> dict[str, float | bool]:
    # time.sleep(torch.abs(torch.randn(1)).item() / kSLEEP_CONSTANT)
    max_tokens = 16384
    messages_structure = [
            {"role": "system", "content": f"{DATA_GENERATION_SYSTEM_MESSAGE_WITH_EXPLANATION}"},
            {"role": "user",
             "content": f"{query_and_response_string}"}
        ]
    completion = client.beta.chat.completions.parse(
        model=LOG_PROB_MODEL,
        messages=messages_structure,
        response_format=IncorrectSolutionGeneratorWithConfidenceAndExplanation,
        max_completion_tokens=max_tokens,
        logprobs=False,
        temperature=0.0,
        user="sdm_llm_branching_v1",
        seed=0
    )
    response_object = completion.choices[0].message.parsed
    return response_object



def llm_api_controller(query_and_response_string: str):
    call_schedule = 0
    while True:
        try:
            response_object = create_negative(query_and_response_string)
            return response_object
        except:
            if call_schedule == 4:
                response_object = IncorrectSolutionGeneratorWithConfidenceAndExplanation(
                    correct_solution_is_correct_classification=False,
                    confidence_in_classification=0.01,
                    short_explanation_for_classification_confidence="ERROR",
                    incorrect_solution="ERROR",
                    short_explanation_why_new_incorrect_solution_is_wrong="ERROR"
                )
                # additional final wait
                call_schedule += 2
                exception_backoff = 2 ** call_schedule + (torch.abs(torch.randn(1)).item() * 30)
                time.sleep(exception_backoff)
                return response_object
            exception_backoff = 2 ** call_schedule + torch.abs(torch.randn(1)).item()
            time.sleep(exception_backoff)
            call_schedule += 1


def construct_question_and_original_solution(json_list, instance_index):
    instance = json_list[instance_index]
    question_from_conversation = instance["question_from_conversation"]
    provided_solution = instance["provided_solution"]
    original_line_id = instance["original_line_id"]
    query_and_response_string = f"<question> {question_from_conversation} </question> <correct_solution> {provided_solution} </correct_solution>"
    return original_line_id, query_and_response_string, question_from_conversation, provided_solution


def get_existing_ids(filepath_with_name):
    existing_ids = set()
    with codecs.open(filepath_with_name, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            json_obj = json.loads(line)
            existing_ids.add(json_obj["original_line_id"])
    return existing_ids


def construct_negative_and_save_shard(indexes_as_list, json_list, output_file):
    time.sleep(torch.abs(torch.randn(1)).item() / kSLEEP_CONSTANT)
    if Path(output_file).exists():
        existing_ids = get_existing_ids(output_file)
    else:
        existing_ids = set()

    for row_index in indexes_as_list:
        if row_index > len(json_list) - 1:
            break
        if json_list[row_index]["original_line_id"] in existing_ids:
            continue
        original_line_id, query_and_response_string, question_from_conversation, provided_solution = \
            construct_question_and_original_solution(json_list, row_index)
        response_object = llm_api_controller(query_and_response_string=query_and_response_string)

        json_obj = {
            "correct_solution_is_correct_classification": response_object.correct_solution_is_correct_classification,
            "confidence_in_classification": response_object.confidence_in_classification,
            "short_explanation_for_classification_confidence": response_object.short_explanation_for_classification_confidence,
            "incorrect_solution": response_object.incorrect_solution,
            "short_explanation_why_new_incorrect_solution_is_wrong": response_object.short_explanation_why_new_incorrect_solution_is_wrong
        }
        json_obj["original_line_id"] = original_line_id
        json_obj["question_from_conversation"] = question_from_conversation
        json_obj["provided_solution"] = provided_solution
        data_utils.save_by_appending_json_lines(output_file, [json_obj])
        existing_ids.add(original_line_id)
    print(f"Lines {indexes_as_list[0]}-{indexes_as_list[-1]} complete.")


async def run_tasks_dynamically(task_configs: List[Tuple[Callable, Tuple[Any, ...]]]):
    """
    Dynamically run a list of tasks concurrently.

    Args:
        task_configs: A list of tuples, each containing:
            - A function to run
            - A tuple of arguments to pass to the function

    Returns:
        List of results from all tasks
    """
    tasks = []

    # Create tasks dynamically
    for func, args in task_configs:
        task = asyncio.to_thread(func, *args)
        tasks.append(task)

    # Run all tasks concurrently
    await asyncio.gather(*tasks)


async def main(options, json_list):

    # Create a list of task configurations dynamically
    task_configs = []
    if options.start_index < len(json_list):  # invalid start/end are safely ignored to simplify bash scripts
        total_size = len(json_list[options.start_index: options.start_index+options.total_lines])
        if total_size > 0:
            row_indexes = np.arange(options.start_index, options.start_index+total_size)
            for np_index_list_for_shard in np.array_split(row_indexes, options.shards):
                indexes_as_list = [int(x) for x in np_index_list_for_shard.tolist()]
                if len(indexes_as_list) > 0:
                    output_file = os.path.join(options.output_dir, f"openthoughts_synthetic_neg_{indexes_as_list[0]}_{indexes_as_list[-1]}.jsonl")
                    task_configs.append((construct_negative_and_save_shard, (indexes_as_list, json_list, output_file)))
            if len(task_configs) > 0:
                print(f"Running {len(task_configs)} tasks concurrently...")
                await run_tasks_dynamically(task_configs)

    print(f"All tasks completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="-----[Generate synthetic negatives via an LLM]-----")
    parser.add_argument("--input_file", default="", help="")
    parser.add_argument("--start_index", default=0, type=int, help="")
    parser.add_argument("--total_lines", default=1000, type=int, help="")
    parser.add_argument("--shards", default=10, type=int, help="")
    parser.add_argument("--output_dir", default="", help="")
    options = parser.parse_args()

    print("In practice to save costs and time, "
          "it is recommended that you upload shards of the data for batch prediction. "
          "Consult the Azure/OpenAI documentation for details. This is only provided for reference on the "
          "parameters used. Exiting.")
    exit()
    start_time = time.time()
    time.sleep(torch.abs(torch.randn(1)).item())
    json_list = data_utils.read_jsons_lines_file(options.input_file)
    print("Read input")
    asyncio.run(main(options, json_list))
    cumulative_time = time.time() - start_time
    print(f"Cumulative running time: {cumulative_time}")
