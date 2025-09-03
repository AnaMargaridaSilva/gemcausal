import os
import re
from typing import Any, Union, Optional
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict, Value, load_dataset
from enum import Enum

from .. import (
    DatasetType,
    NumCausalType,
    PlicitType,
    SentenceType,
    SpanTags,
    SpanTagsFormat,
    TaskType,
)
from ..setting import tpl_task_explicit, tpl_task_implicit
from .split_dataset import filter_plicit_dataset


def cast_column_to_int(ds: Dataset, column: str) -> Dataset:
    features = ds.features.copy()
    features[column] = Value("int64")
    ds = ds.cast(features)
    return ds


def remove_tag(tok: str) -> str:
    # Remove all other tags: E.g., <SIG0>, <SIG1>...
    return re.sub(r"</*[A-Z]+\d*>", "", tok)


def get_bio(text_w_pairs: str) -> tuple[list[str], list[str]]:
    tokens: list[str] = []
    ce_tags: list[str] = []
    next_tag: str = "O"
    tag: str = "O"

    for tok in text_w_pairs.split(" "):
        if "<ARG0>" in tok:
            tok = tok.replace("<ARG0>", "")
            tag = "B-C"
            next_tag = "I-C"
        elif "</ARG0>" in tok:
            tok = tok.replace("</ARG0>", "")
            tag = "I-C"
            next_tag = "O"
        elif "<ARG1>" in tok:
            tok = tok.replace("<ARG1>", "")
            tag = "B-E"
            next_tag = "I-E"
        elif "</ARG1>" in tok:
            tok = tok.replace("</ARG1>", "")
            tag = "I-E"
            next_tag = "O"

        tokens.append(remove_tag(tok))
        ce_tags.append(tag)
        tag = next_tag

    return tokens, ce_tags


def get_bio_for_datasets(example: dict[str, Any]) -> dict[str, Any]:
    tokens: list[str]
    tags: list[str]
    tokens, tags = get_bio(example["causal_text_w_pairs"])
    example["tokens"] = tokens
    example["tags"] = tags
    # Tagged text with SpanTags for output
    example["tagged_text"] = (
        example["causal_text_w_pairs"]
        .replace("<ARG0>", SpanTags.cause_begin)
        .replace("</ARG0>", SpanTags.cause_end)
        .replace("<ARG1>", SpanTags.effect_begin)
        .replace("</ARG1>", SpanTags.effect_end)
    )
    return example


def extract_all_causes_effects(text: str) -> tuple[list[str], list[str]]:
    cause_pattern = re.compile(r"<ARG0>(.*?)</ARG0>")
    effect_pattern = re.compile(r"<ARG1>(.*?)</ARG1>")
    causes = cause_pattern.findall(text)
    effects = effect_pattern.findall(text)
    return causes, effects


def _filter_data_by_num_sent(
    ds: Dataset, dataset_enum: Union[str, Enum], sentencetype_enum: Enum
) -> Dataset:
    if (
        dataset_enum in (DatasetType.altlex, DatasetType.because, DatasetType.semeval)
        and sentencetype_enum != SentenceType.all
    ):
        raise ValueError(f"filter_num_sent is not supported for {dataset_enum}")
    if sentencetype_enum == SentenceType.intra:
        ds = ds.filter(lambda x: x["num_sents"] == 1)
    elif sentencetype_enum == SentenceType.inter:
        ds = ds.filter(lambda x: x["num_sents"] >= 2)
    return ds


def _filter_data_by_num_causal(ds: Dataset, numcausal_enum: Enum) -> Dataset:
    df: pd.DataFrame = ds.to_pandas()
    if numcausal_enum == NumCausalType.single:
        df = df[~df.duplicated(subset=["corpus", "doc_id", "sent_id"], keep=False)]
    elif numcausal_enum == NumCausalType.multi:
        df = df[df.duplicated(subset=["corpus", "doc_id", "sent_id"], keep=False)]
    return Dataset.from_pandas(df, preserve_index=False)


def _filter_data_by_plicit(
    ds: Dataset, dataset_enum: Enum, plicit_enum: Enum
) -> Dataset:
    if plicit_enum == PlicitType.explicit:
        assert dataset_enum in tpl_task_explicit
        ds = filter_plicit_dataset(ds, plicit_enum)
    elif plicit_enum == PlicitType.implicit:
        assert dataset_enum in tpl_task_implicit
        ds = filter_plicit_dataset(ds, plicit_enum)
    else:
        assert plicit_enum == PlicitType.all
    return ds


def _load_data_CNC_sequence_classification(data_path: str) -> Dataset:
    ds: Dataset = load_dataset("csv", data_files=data_path, split="train")

    # Derive label from num_rs
    ds = ds.map(lambda x: {"labels": 0 if x["num_rs"] == 0 else 1})
    # Apply BIO tagging
    ds = ds.map(get_bio_for_datasets)
    # Rename for consistency
    ds = ds.rename_column("eg_id", "example_id")
    return ds


def _load_data_CNC_span_detection(data_path: str) -> Dataset:
    df: pd.DataFrame = pd.read_csv(data_path)
    # Keep only rows where num_rs > 0
    df = df[df["num_rs"] > 0]
    ds: Dataset = Dataset.from_pandas(df)
    ds = ds.map(get_bio_for_datasets)
    ds = ds.rename_column("eg_id", "example_id")
    return ds


def _filter_data(
    ds: Dataset,
    task_enum: Enum,
    dataset_enum: Enum,
    numcausal_enum: Enum,
    sentencetype_enum: Enum,
    plicit_enum: Enum,
) -> Dataset:
    if task_enum == TaskType.span_detection:
        ds = _filter_data_by_num_causal(ds, numcausal_enum=numcausal_enum)
    ds = _filter_data_by_num_sent(
        ds, dataset_enum=dataset_enum, sentencetype_enum=sentencetype_enum
    )
    if task_enum == TaskType.span_detection:
        ds = _filter_data_by_plicit(
            ds, dataset_enum=dataset_enum, plicit_enum=plicit_enum
        )
    return ds


def load_data_CNC(
    dataset_enum: Enum,
    task_enum: Enum,
    sentencetype_enum: Enum,
    numcausal_enum: Enum,
    plicit_enum: Enum,
    data_dir: str,
    seed: int,
) -> tuple[Dataset, Dataset, Dataset]:
    train_val_data_path = os.path.join(data_dir, "train_subtask2_grouped.csv")
    test_data_path = os.path.join(data_dir, "dev_subtask2_grouped.csv")

    if task_enum == TaskType.sequence_classification:
        ds_train_val = _load_data_CNC_sequence_classification(train_val_data_path)
        ds_test = _load_data_CNC_sequence_classification(test_data_path)
    elif task_enum == TaskType.span_detection:
        ds_train_val = _load_data_CNC_span_detection(train_val_data_path)
        ds_test = _load_data_CNC_span_detection(test_data_path)

    ds_test = _filter_data(
        ds_test,
        task_enum=task_enum,
        dataset_enum=dataset_enum,
        numcausal_enum=numcausal_enum,
        sentencetype_enum=sentencetype_enum,
        plicit_enum=plicit_enum,
    )

    if task_enum == TaskType.sequence_classification:
        ds_train_val = cast_column_to_int(ds_train_val, "labels")
        ds_test = cast_column_to_int(ds_test, "labels")

    valid_size = len(ds_test) if len(ds_test) * 4 < len(ds_train_val) else int(len(ds_train_val) * 0.2)
    dsd_train_val: DatasetDict = ds_train_val.train_test_split(
        test_size=valid_size, shuffle=True, seed=seed
    )
    ds_train = dsd_train_val["train"]
    ds_valid = dsd_train_val["test"]

    return DatasetDict({
            "train": ds_train,
            "valid": ds_valid,
            "test": ds_test
        })
