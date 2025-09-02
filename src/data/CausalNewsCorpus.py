import os
import re
from typing import Any, Optional, Union
from enum import Enum

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, Value, load_dataset

from .split_dataset import filter_plicit_dataset
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


def cast_column_to_int(ds: Dataset, column: str) -> Dataset:
    features = ds.features.copy()
    features[column] = Value("int64")
    ds = ds.cast(features)
    return ds


def remove_tag(tok):
    # Remove all other tags: E.g. <SIG0>, <SIG1>...
    return re.sub(r"</*[A-Z]+\d*>", "", tok)


def get_bio(text_w_pairs: str) -> tuple[list[str], list[str]]:
    tokens: list[str] = []
    ce_tags: list[str] = []
    next_tag: str = "O"
    tag: str = "O"
    for tok in text_w_pairs.split(" "):
        # Replace if special
        if "<ARG0>" in tok:
            tok = re.sub("<ARG0>", "", tok)
            tag = "B-C"
            next_tag = "I-C"
        elif "</ARG0>" in tok:
            tok = re.sub("</ARG0>", "", tok)
            tag = "I-C"
            next_tag = "O"
        elif "<ARG1>" in tok:
            tok = re.sub("<ARG1>", "", tok)
            tag = "B-E"
            next_tag = "I-E"
        elif "</ARG1>" in tok:
            tok = re.sub("</ARG1>", "", tok)
            tag = "I-E"
            next_tag = "O"

        tokens.append(remove_tag(tok))
        ce_tags.append(tag)
        tag = next_tag
    return tokens, ce_tags


def get_bio_for_datasets(example: dict[str, Any]) -> dict[str, Any]:
    tokens: list[str]
    tags: list[str]
    tokens, tags = get_bio(example["text_w_pairs"])
    example["tokens"] = tokens
    example["tags"] = tags
    example["tagged_text"] = (
        example["text_w_pairs"]
        .replace("<ARG0>", SpanTags.cause_begin)
        .replace("</ARG0>", SpanTags.cause_end)
        .replace("<ARG1>", SpanTags.effect_begin)
        .replace("</ARG1>", SpanTags.effect_end)
    )
    return example


def custom_agg(group: Any) -> pd.Series:
    reg_cause: re.Pattern = re.compile(
        f"{SpanTags.cause_begin}(.*?){SpanTags.cause_end}"
    )
    reg_effect: re.Pattern = re.compile(
        f"{SpanTags.effect_begin}(.*?){SpanTags.effect_end}"
    )
    result: dict[str, Union[str, int, None]] = {}
    for col in group.columns:
        if col == "tagged_text":
            text: Optional[str] = group["text"].iloc[0]
            for i, idx in enumerate(
                np.argsort(
                    [
                        -(x.find(SpanTags.cause_begin))
                        for x in group["text_w_pairs"].tolist()
                    ]
                )
            ):
                cause: str = re.search(reg_cause, group[col].iloc[idx]).group(1)
                effect: str = re.search(reg_effect, group[col].iloc[idx]).group(1)
                if cause in text:
                    text = text.replace(
                        cause,
                        SpanTagsFormat.cause_begin.format(i + 1)
                        + cause
                        + SpanTagsFormat.cause_end.format(i + 1),
                    )
                else:
                    text = None
                    break
                if effect in text:
                    text = text.replace(
                        effect,
                        SpanTagsFormat.effect_begin.format(i + 1)
                        + effect
                        + SpanTagsFormat.effect_end.format(i + 1),
                    )
                else:
                    text = None
                    break
            result[col] = text
        else:
            result[col] = group[col].iloc[0]
    return pd.Series(result)


def _filter_data_by_num_sent(
    ds: Dataset, dataset_enum: Enum, sentencetype_enum: Enum
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
    else:
        if numcausal_enum == NumCausalType.multi:
            df = df[df.duplicated(subset=["corpus", "doc_id", "sent_id"], keep=False)]
        else:
            assert numcausal_enum == NumCausalType.all
        groups = df.groupby(["corpus", "doc_id", "sent_id"])
        df = groups.apply(custom_agg).reset_index(drop=True)
        # Drop nested causal with primitive way
        df.dropna(subset=["tagged_text"], inplace=True)
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
    # Derive labels from num_rs
    # if num_rs == 0 → label 0
    # if num_rs > 0 → label 1
    def assign_label(example):
        example["labels"] = 0 if example["num_rs"] == 0 else 1
        return example

    ds = ds.map(assign_label)

    # Preprocess with BIO tagging
    ds = ds.map(get_bio_for_datasets)

    # Rename for consistency
    ds = ds.rename_column("eg_id", "example_id")

    return ds

def _load_data_CNC_span_detection(data_path: str) -> Dataset:
    df: pd.DataFrame = pd.read_csv(data_path)
    # Keep only rows where num_rs > 0
    df = df[df["num_rs"] > 0]
    ds: Dataset = Dataset.from_pandas(df)
    # Preprocess with BIO tagging
    ds = ds.map(get_bio_for_datasets)
    # Rename for consistency
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

    # dataset_path_prefix: str
  
    # train_val_data_path: str = os.path.join(
        data_dir, f"{dataset_path_prefix}_train.csv"
    )
    # test_data_path: str = os.path.join(data_dir, f"{dataset_path_prefix}_test.csv")

    train_file = os.path.join(data_dir, "train_subtask2_grouped.csv")
    test_file = os.path.join(data_dir, "dev_subtask2_grouped.csv")

    ds_train_val: Dataset
    ds_test: Dataset
  
    if task_enum == TaskType.sequence_classification:
        ds_train_val = _load_data_CNC_sequence_classification(
            train_val_data_path
        )
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
    valid_size: int
    if len(ds_test) * 4 < len(ds_train_val):
        valid_size = len(ds_test)
    else:
        valid_size = int(len(ds_train_val) * 0.2)
    dsd_train_val: DatasetDict = ds_train_val.train_test_split(
        test_size=valid_size, shuffle=True, seed=seed
    )
    ds_train = dsd_train_val["train"]
    ds_valid = dsd_train_val["test"]
    return ds_train, ds_valid, ds_test
