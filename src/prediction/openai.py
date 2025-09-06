import datetime
import time
import itertools
import json
import os
import random
import re
from argparse import Namespace
from enum import Enum
from typing import Any, Dict, List, Tuple

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

# -------------------- Helper functions -------------------- #

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

def remove_signal_tags(text: str) -> str:
    """
    Remove any <SIG...>...</SIG...> tags from a text string.
    """
    return re.sub(r"<SIG\d+>(.*?)<\/?SIG\d+>", r"\1", text)

# --- Extraction helpers --- #

def extract_pairs_with_mark(text: str) -> List[Tuple[str, str]]:
    """
    Extract pairs from model output (using <c>...</c> and <e>...</e>).
    """
    causes = [m.strip() for m in re.findall(r"<c>(.*?)</c>", text, flags=re.DOTALL)]
    effects = [m.strip() for m in re.findall(r"<e>(.*?)</e>", text, flags=re.DOTALL)]
    return list(zip(causes, effects))

def extract_all_causes_effects(text_w_pairs: str) -> Tuple[List[str], List[str]]:
    """
    Extract causes/effects from gold data (using <ARG0> and <ARG1>).
    """
    cause_pattern = re.compile(r"<ARG0>(.*?)</ARG0>")
    effect_pattern = re.compile(r"<ARG1>(.*?)</ARG1>")
    causes = [m.group(1).strip() for m in cause_pattern.finditer(text_w_pairs)]
    effects = [m.group(1).strip() for m in effect_pattern.finditer(text_w_pairs)]
    return causes, effects

def split_relation(rel: str) -> Tuple[str, str]:
    """
    Extract cause/effect spans from inline tagged relation text.
    Example: 'Relation1: The <c>rain</c> caused <e>flooding</e>.'
    """
    cause_match = re.search(r"<c>(.*?)</c>", rel)
    effect_match = re.search(r"<e>(.*?)</e>", rel)
    cause = remove_signal_tags(cause_match.group(1).strip()) if cause_match else ""
    effect = remove_signal_tags(effect_match.group(1).strip()) if effect_match else ""
    return cause, effect

# --- Model helper --- #

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(model: str, prompt: str):
    model_obj = genai.GenerativeModel(model)
    response = model_obj.generate_content(prompt, generation_config={"temperature": 0})
    return response

def remove_marks(text: str) -> str:
    extracted = re.findall(r"\[([^\]]+)\]", text)
    return " ; ".join(extracted) if extracted else text

def compute_f1_score(true_span: str, pred_span: str) -> Tuple[float, float, float]:
    true_tokens, pred_tokens = set(true_span.split()), set(pred_span.split())
    tp, fp, fn = len(true_tokens & pred_tokens), len(pred_tokens - true_tokens), len(true_tokens - pred_tokens)
    if not true_tokens and not pred_tokens:
        return 1.0, 1.0, 1.0
    if tp == 0:
        return 0.0, 0.0, 0.0
    precision, recall = tp / (tp + fp), tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1

# -------------------- Main function -------------------- #

def predict(args: Namespace) -> None:
    api_key_validation()
    task_enum: Enum = TaskType[args.task_type]
    dataset_enum: Enum = DatasetType[args.dataset_type]
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

    # Load dataset
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

    annotation: str = template["header_example"]
    if task_enum == TaskType.span_detection:
        wants_relations = (
            "Relation" in template["task_description"]
            or "Relation" in template.get("format_class", "")
        )
        for i in range(shot):
            annotation += template["format_text"].format(dsd_icl[i]["text"])
            if wants_relations:
                pairs = extract_pairs_with_mark(dsd_icl[i]["tagged_text"])
                for j, (c, e) in enumerate(pairs, 1):
                    annotation += f"Relation{j}: <c>{c}</c> <e>{e}</e>\n"
                annotation += "\n"
            else:
                cause_spans, effect_spans = extract_all_causes_effects(dsd_icl[i]["tagged_text"])
                annotation += template["format_class"].format(cause_spans, effect_spans)

        def format_prompt(example: dict[str, Any]) -> dict[str, Any]:
            prompt: str = template["task_description"]
            if shot > 0:
                prompt += annotation
            prompt += template["question"] + template["format_text"].format(example["text"])
            example["prompt"] = prompt
            return example

    ds_test: Dataset = dsd["test"]
    ds_test = ds_test.map(format_prompt)

    # -------------------- Inference -------------------- #
    logger.info("Inference starts")
    batch_size = 2
    lst_output: list[str] = []

    for i in tqdm(range(0, len(ds_test), batch_size)):
        batch_prompts = ds_test["prompt"][i:i + batch_size]
        batch_results = [completion_with_backoff(model, p).text for p in batch_prompts]
        lst_output.extend(batch_results)
        if i + batch_size < len(ds_test):
            time.sleep(60)

    logger.info("Inference ends")
    ds_test = ds_test.add_column("output", lst_output)

    # -------------------- Extract predicted/true relations -------------------- #
    def extract_span(example: Dict[str, Any]) -> Dict[str, Any]:
        # Gold
        true_causes, true_effects = extract_all_causes_effects(example["causal_text_w_pairs"])
        true_causes = [remove_signal_tags(c) for c in true_causes]
        true_effects = [remove_signal_tags(e) for e in true_effects]
        true_relations = [
            f"Relation{i+1}: <c>{c}</c> <e>{e}</e>"
            for i, (c, e) in enumerate(zip(true_causes, true_effects))
        ]

        # Predictions
        pred_relations = [
            remove_signal_tags(line.strip())
            for line in example["output"].splitlines()
            if line.strip()
        ]

        example["true_relations"] = true_relations
        example["pred_relations"] = pred_relations
        return example

    ds_output = ds_test.map(extract_span)

    # -------------------- Compute metrics -------------------- #
    total_relation_exact = 0
    all_true_causes, all_pred_causes = [], []
    all_true_effects, all_pred_effects = [], []

    for i in range(len(ds_output)):
        true_rel = ds_output[i]["true_relations"]
        pred_rel = ds_output[i]["pred_relations"]
        total_relation_exact += len(set(true_rel) & set(pred_rel))

        t_c, t_e = zip(*map(split_relation, true_rel)) if true_rel else ([], [])
        p_c, p_e = zip(*map(split_relation, pred_rel)) if pred_rel else ([], [])

        all_true_causes.extend(t_c)
        all_true_effects.extend(t_e)
        all_pred_causes.extend(p_c)
        all_pred_effects.extend(p_e)

    # Compute cause/effect F1
    cause_metrics = [compute_f1_score(t, p) for t, p in zip(all_true_causes, all_pred_causes)]
    effect_metrics = [compute_f1_score(t, p) for t, p in zip(all_true_effects, all_pred_effects)]

    result = {}
    result["relation_exact_match"] = total_relation_exact
    result["cause_precision"], result["cause_recall"], result["cause_f1"] = map(
        lambda x: sum(x) / len(x), zip(*cause_metrics)
    )
    result["effect_precision"], result["effect_recall"], result["effect_f1"] = map(
        lambda x: sum(x) / len(x), zip(*effect_metrics)
    )
    result["precision"] = (result["cause_precision"] + result["effect_precision"]) / 2
    result["recall"] = (result["cause_recall"] + result["effect_recall"]) / 2
    result["f1"] = (result["cause_f1"] + result["effect_f1"]) / 2

    # -------------------- Save -------------------- #
    ds_output = ds_output.remove_columns(
        [c for c in ds_test.column_names if c not in ["example_id","text","tagged_text","output","true_relations","pred_relations"]]
    )

    logger.info("Result: %s", result)
    result = {**result, **{
        "task_type": args.task_type,
        "dataset_type": args.dataset_type,
        "intra-/inter-sent": filter_num_sent,
        "single-/multi-causal": filter_num_causal,
        "ex-/im-plicit": filter_plicit_type,
        "model": model,
        "template": args.template,
        "shot": shot,
        "seed": seed,
    }}

    filehead = datetime.datetime.now().strftime("%Y%m%d_%H%M_") + f"{args.task_type}_{args.dataset_type}"
    ds_output.to_csv(os.path.join(output_dir, f"{filehead}.csv"))
    ds_output.to_json(os.path.join(output_dir, f"{filehead}_predictions.json"))
    with open(os.path.join(output_dir, f"{filehead}.json"), "w") as f:
        json.dump(result, f, indent=4, sort_keys=True)
