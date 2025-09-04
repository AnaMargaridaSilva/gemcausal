# 04092025: a funcionar se quisermos juntar: 
# true_causes separadas por ;
# true_effects separados por ; 
# pred_causes separados por ; 
# pred_effects separados por ;

# Para o prompt alterado 

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

def extract_spans_with_mark(text: str) -> Tuple[str, str]:
    cause_marker_pattern = r"<c(\d*)>(.*?)</c\1>"
    effect_marker_pattern = r"<e(\d*)>(.*?)</e\1>"
    extract = lambda pattern: " ".join(
        [f"[{match.group(2).strip()}]" for match in re.finditer(pattern, text)]
    )
    return extract(cause_marker_pattern), extract(effect_marker_pattern)

def extract_pairs_with_mark(text: str) -> List[Tuple[str, str]]:
    """Return [(cause, effect), ...] by pairing tags that share the same numeric id."""
    tagged = re.findall(r"<([ce])(\d*)>(.*?)</\1\2>", text, flags=re.DOTALL)
    by_id: Dict[str, Dict[str, str]] = {}
    for tag, idx, span in tagged:
        k = idx if idx else "0"
        by_id.setdefault(k, {})
        by_id[k][tag] = span.strip()
    pairs: List[Tuple[str, str]] = []
    def _key(x: str) -> int:
        try: return int(x)
        except: return 0
    for k in sorted(by_id.keys(), key=_key):
        c = by_id[k].get("c")
        e = by_id[k].get("e")
        if c and e:
            pairs.append((c, e))
    return pairs

def extract_all_causes_effects(text_w_pairs: str) -> Tuple[List[str], List[str]]:
    cause_pattern = re.compile(r"<ARG0>(.*?)</ARG0>")
    effect_pattern = re.compile(r"<ARG1>(.*?)</ARG1>")
    causes = [m.group(1).strip() for m in cause_pattern.finditer(text_w_pairs)]
    effects = [m.group(1).strip() for m in effect_pattern.finditer(text_w_pairs)]
    return causes, effects

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(model: str, prompt: str):
    model_obj = genai.GenerativeModel(model)
    response = model_obj.generate_content(prompt, generation_config={"temperature": 0})
    return response

def compute_metrics(
    y_true: list[str], y_pred: list[str], labels: list[str], average: str
) -> dict[str, float]:
    assert average in {"macro", "micro", "binary"}, f"average {average} not implemented"
    if average == "binary":
        assert len(labels) == 1, "In binary classification the number of labels must be 1"
        average = None
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average, labels=labels
    )
    if not isinstance(precision, float):
        precision = precision[0]
    if not isinstance(recall, float):
        recall = recall[0]
    if not isinstance(f1, float):
        f1 = f1[0]
    result: Dict[str, float] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }
    arr = confusion_matrix(y_true, y_pred, normalize="true")
    for i in range(len(arr)):
        result[f"accuracy_{i}"] = arr[i, i]
    return result

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
                    annotation += f"Relation{j}: [{c}] [{e}]\n"
                annotation += "\n"
            else:
                cause_spans, effect_spans = extract_spans_with_mark(dsd_icl[i]["tagged_text"])
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

    # -------------------- Extract predicted/true spans -------------------- #
    def extract_span(example: Dict[str, Any]) -> Dict[str, Any]:
        lines = [ln.strip() for ln in example["output"].splitlines() if ln.strip()]
        pred_causes: List[str] = []
        pred_effects: List[str] = []
        for line in lines:
            if line.startswith("Causes:"):
                pred_causes = re.findall(r"\[([^\]]+)\]", line)
            elif line.startswith("Effects:"):
                pred_effects = re.findall(r"\[([^\]]+)\]", line)
            elif line.lower().startswith("relation"):
                parts = re.findall(r"\[([^\]]+)\]", line)
                if len(parts) >= 2:
                    pred_causes.append(parts[0].strip())
                    pred_effects.append(parts[1].strip())
        true_causes, true_effects = extract_all_causes_effects(example["causal_text_w_pairs"])
        example["true_cause"] = " ; ".join(true_causes)
        example["true_effect"] = " ; ".join(true_effects)
        example["pred_cause"] = " ; ".join(pred_causes)
        example["pred_effect"] = " ; ".join(pred_effects)
        return example

    logger.info(f"Columns before mapping: {ds_test.column_names}")
    ds_output = ds_test.map(extract_span)

    # -------------------- Compute metrics -------------------- #
    result = {}
    results_cause = [
        t.strip() == p.strip()
        for t, p in zip(ds_output["true_cause"], ds_output["pred_cause"])
    ]
    results_effect = [
        t.strip() == p.strip()
        for t, p in zip(ds_output["true_effect"], ds_output["pred_effect"])
    ]
    total_matches = sum(results_cause) + sum(results_effect)
    total_pairs = len(results_cause) + len(results_effect)
    accuracy = total_matches / total_pairs if total_pairs > 0 else 0.0
    logger.info("Cause match count: %d / %d", sum(results_cause), len(results_cause))
    logger.info("Effect match count: %d / %d", sum(results_effect), len(results_effect))
    logger.info("Overall accuracy: %.2f%%", accuracy * 100)
    cause_metrics = [compute_f1_score(t, p) for t, p in zip(ds_output["true_cause"], ds_output["pred_cause"])]
    effect_metrics = [compute_f1_score(t, p) for t, p in zip(ds_output["true_effect"], ds_output["pred_effect"])]
    result["cause_precision"], result["cause_recall"], result["cause_f1"] = map(
        lambda x: sum(x) / len(x), zip(*cause_metrics)
    )
    result["effect_precision"], result["effect_recall"], result["effect_f1"] = map(
        lambda x: sum(x) / len(x), zip(*effect_metrics)
    )
    result["precision"] = (result["cause_precision"] + result["effect_precision"]) / 2
    result["recall"] = (result["cause_recall"] + result["effect_recall"]) / 2
    result["f1"] = (result["cause_f1"] + result["effect_f1"]) / 2
    ds_output = ds_output.remove_columns(
        [c for c in ds_test.column_names if c not in ["example_id","text","tagged_text","output","true_cause","true_effect","pred_cause","pred_effect"]]
    )

    # -------------------- Save -------------------- #
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

