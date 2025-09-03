import datetime
import time
import itertools
import json
import os
import random
import re
from argparse import Namespace
from enum import Enum
from typing import Any, Dict, List, Set, Tuple, Union

import datasets
import numpy as np
import google.generativeai as genai
from datasets import Dataset, DatasetDict
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tqdm import tqdm


from .. import (
    DatasetType,
    NumCausalType,
    PlicitType,
    SentenceType,
    TaskType,
    assert_dataset_task_pair,
    logger,
)
from ..data.load_data import load_data
from ..setting import assert_filter_option

def api_key_validation() -> None:
    api_key: str = os.environ["GEMINI_API_KEY"]
    assert api_key != "", "Environment variable GEMINI_API_KEY must be set to use Gemini models"
    genai.configure(api_key=api_key)


def read_template(path: str) -> dict[str, str]:
    with open(path, "r") as f:
        template: dict[str, str] = json.load(f)
    required_keys: set[str] = {
        "task_description",
        "header_example",
        "format_text",
        "format_class",
        "question",
    }
    left_keys: set[str] = required_keys - set(template.keys())
    assert len(left_keys) == 0, f"Following keys are not in template: {left_keys}"
    return template



# Updated 
def extract_spans_with_mark(text_w_pairs: List[str]) -> Tuple[str, str]:
    """
    Extracts all <ARG0> and <ARG1> spans from a list of causal text pairs.
    Returns them as space-separated strings of causes and effects.
    """
    causes = []
    effects = []
    for pair in text_w_pairs:
        causes.extend(re.findall(r"<ARG0>(.*?)</ARG0>", pair))
        effects.extend(re.findall(r"<ARG1>(.*?)</ARG1>", pair))
    return " ".join(causes), " ".join(effects)

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(model: str, prompt: str):
    model_obj = genai.GenerativeModel(model)
    response = model_obj.generate_content(prompt, generation_config={"temperature": 0})
    return response




# Updated 
def extract_span(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts true and predicted causes/effects for a test example.
    """
    # Extract true spans from your grouped column
    true_cause_spans, true_effect_spans = extract_spans_with_mark(
        example.get("causal_text_w_pairs", [])
    )
    example["true_cause"] = true_cause_spans
    example["true_effect"] = true_effect_spans

    # Extract predicted spans if 'output' exists
    """
    if "output" in example:
        output_text = example["output"]
        extracted_spans = re.findall(r"\[([^\]]+)\]", output_text)
        # Split predicted spans into causes and effects
        num_causes = len(true_cause_spans.split())
        num_effects = len(true_effect_spans.split())
        example["pred_cause"] = " ".join(extracted_spans[:num_causes]) if extracted_spans else ""
        example["pred_effect"] = " ".join(extracted_spans[num_causes:num_causes+num_effects]) if extracted_spans else ""

    return example
    """
    if "output" in example:
        output_text = example["output"]
        extracted_spans = re.findall(r"\[([^\]]+)\]", output_text)
        if len(extracted_spans) >= 2:
            mid = len(extracted_spans) // 2
            example["pred_cause"] = " ".join(extracted_spans[:mid])
            example["pred_effect"] = " ".join(extracted_spans[mid:])
        else:
            example["pred_cause"] = ""
            example["pred_effect"] = ""
    return example


# -------------------------------
# Updated F1 computation (kept separate)
# -------------------------------
def compute_f1_score(true_span: str, pred_span: str) -> Tuple[float, float, float]:
    true_tokens = set(true_span.split())
    pred_tokens = set(pred_span.split())
    
    tp = len(true_tokens & pred_tokens)
    fp = len(pred_tokens - true_tokens)
    fn = len(true_tokens - pred_tokens)
    
    if not true_tokens and not pred_tokens:
        return 1.0, 1.0, 1.0
    if tp == 0:
        return 0.0, 0.0, 0.0
    
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    return precision, recall, f1


# -------------------------------
# Evaluation function with original logic
# -------------------------------
def compute_metrics_for_spans(ds_output: Dataset) -> Dict[str, float]:
    results = {
        "exact_match": 0,
        "cause_precision": 0,
        "cause_recall": 0,
        "cause_f1": 0,
        "effect_precision": 0,
        "effect_recall": 0,
        "effect_f1": 0,
    }
    
    exact_matches = []
    cause_metrics = []
    effect_metrics = []
    


    for true_cause, pred_cause, true_effect, pred_effect in zip(
        ds_output["true_cause"], ds_output["pred_cause"],
        ds_output["true_effect"], ds_output["pred_effect"]
    ):
        exact_matches.append(true_cause.strip() == pred_cause.strip() and true_effect.strip() == pred_effect.strip())
        cause_metrics.append(compute_f1_score(true_cause, pred_cause))
        effect_metrics.append(compute_f1_score(true_effect, pred_effect))
    
    results["exact_match"] = sum(exact_matches) / len(exact_matches)
    for i, key in enumerate(["precision", "recall", "f1"]):
        results[f"cause_{key}"] = sum(row[i] for row in cause_metrics) / len(cause_metrics)
        results[f"effect_{key}"] = sum(row[i] for row in effect_metrics) / len(effect_metrics)
    
    results["precision"] = (results["cause_precision"] + results["effect_precision"]) / 2
    results["recall"] = (results["cause_recall"] + results["effect_recall"]) / 2
    results["f1"] = (results["cause_f1"] + results["effect_f1"]) / 2
    
    return results


def predict(args: Namespace) -> None:
    api_key_validation()
    
    task_type: str = args.task_type
    task_enum: Enum = TaskType[task_type]
    dataset_type: str = args.dataset_type
    dataset_enum: Enum = DatasetType[dataset_type]
    model: str = args.model
    shot: int = args.shot
    output_dir: str = args.output_dir
    seed: int = args.seed
    filter_num_sent: str = args.filter_num_sent
    filter_num_causal: str = args.filter_num_causal
    filter_plicit_type: str = args.filter_plicit_type

    os.makedirs(output_dir, exist_ok=True)
    template: dict[str, str] = read_template(args.template)

    assert_dataset_task_pair(dataset_enum=dataset_enum, task_enum=task_enum)
    assert_filter_option(dataset_enum=dataset_enum, args=args)

    # Load and filter dataset
    dsd: DatasetDict = load_data(
        task_enum=task_enum,
        dataset_enum=dataset_enum,
        sentencetype_enum=SentenceType[filter_num_sent],
        numcausal_enum=NumCausalType[filter_num_causal],
        plicit_enum=PlicitType[filter_plicit_type],
        data_dir=args.data_dir,
        test_samples=args.test_samples,
        seed=seed,
    )

    random.seed(seed)
    dsd_icl: Dataset = dsd["train"].select(
        random.sample(range(len(dsd["train"])), k=shot)
    )

    # Build few-shot annotation
    annotation: str = template["header_example"]
    if task_enum == TaskType.span_detection and not args.evaluate_by_word:
        for i in range(shot):
            annotation += template["format_text"].format(dsd_icl[i]["text"])
            cause_spans, effect_spans = extract_spans_with_mark(
                dsd_icl[i]["causal_text_w_pairs"]
            )
            annotation += template["format_class"].format(cause_spans, effect_spans)

        def format_prompt(example: dict[str, Any]) -> dict[str, Any]:
            prompt: str = template["task_description"]
            if shot > 0:
                prompt += annotation
            prompt += template["question"] + template["format_text"].format(example["text"])
            example["prompt"] = prompt
            return example

    else:
        # For other task types, keep your original logic
        raise NotImplementedError("Only span_detection (non-evaluate_by_word) is implemented in this update.")

    # Prepare test dataset
    ds_test: Dataset = dsd["test"]
    ds_test = ds_test.map(format_prompt)

    # Gemini API call
    logger.info("Inference starts")
    batch_size = 2
    lst_output: list[str] = []

    for i in tqdm(range(0, len(ds_test), batch_size)):
        batch_prompts = ds_test["prompt"][i:i + batch_size]
        batch_results = [completion_with_backoff(model, p).text for p in batch_prompts]
        lst_output.extend(batch_results)
        if i + batch_size < len(ds_test):
            logger.info("Waiting 60 seconds before the next batch...")
            time.sleep(60)

    logger.info("Inference ends")
    ds_test = ds_test.add_column("output", lst_output)

    # Extract spans from predictions
    ds_output: Dataset = ds_test.map(extract_span)

    # Compute metrics
    result: dict[str, float] = compute_metrics_for_spans(ds_output)
    logger.info("Result: %s", result)

    # Save results
    filehead = datetime.datetime.now().strftime("%Y%m%d_%H%M_") + f"{args.task_type}_{args.dataset_type}"
    if filter_num_sent != "all":
        filehead += f"_{filter_num_sent}"
    if filter_num_causal != "all":
        filehead += f"_{filter_num_causal}"
    if filter_plicit_type != "all":
        filehead += f"_{filter_plicit_type}"
    filehead += f"_{model}"

    with open(os.path.join(output_dir, f"{filehead}.json"), "w") as f:
        json.dump(result, f, indent=4, sort_keys=True, separators=(",", ": "))

    # -------------------------------
    # Save CSV with correct columns
    # -------------------------------
    required_cols = [
        "example_id",
        "text",
        "tagged_text",
        "output",
        "true_cause",
        "true_effect",
        "pred_cause",
        "pred_effect",
    ]

    for col in required_cols:
        if col not in ds_output.column_names:
            ds_output = ds_output.add_column(col, [""] * len(ds_output))

    ds_output = ds_output.select([ds_output.column_names.index(c) for c in required_cols])
    ds_output.to_csv(os.path.join(output_dir, f"{filehead}.csv"))
