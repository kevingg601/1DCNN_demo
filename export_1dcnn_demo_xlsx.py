# -*- coding: utf-8 -*-
"""Export the existing 1DCNN demo train/test splits as wide XLSX tables.

Each row is one (2048, 2) demo sample.  ``Class_Name`` is the target column,
followed by the Drive End (DE) and Fan End (FE) time-series values.
"""

import argparse
import os
import pickle
import re
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import xlsxwriter


SAMPLE_LENGTH = 2048
CHANNEL_COUNT = 2
EXPECTED_COLUMNS = 1 + SAMPLE_LENGTH * CHANNEL_COUNT
CLASS_NAMES = {
    1: "Normal Baseline",
    2: "0.007_Inner_Race",
    3: "0.014_Ball",
    4: "0.021_Ball",
    5: "0.007_Ball",
    6: "0.014_Inner_Race",
    7: "0.021_Inner_Race",
    8: "0.007_Outer_Race",
    9: "0.014_Outer_Race",
    10: "0.021_Outer_Race",
}
HEADERS = (
    ["Class_Name"]
    + [f"DE_{index:04d}" for index in range(1, SAMPLE_LENGTH + 1)]
    + [f"FE_{index:04d}" for index in range(1, SAMPLE_LENGTH + 1)]
)


def validate_dataset(features, labels, dataset_name):
    """Validate the shape and labels expected by run_1DCNN_demo.py."""
    if features.ndim != 3 or features.shape[1:] != (SAMPLE_LENGTH, CHANNEL_COUNT):
        raise ValueError(
            f"{dataset_name} features must have shape (N, {SAMPLE_LENGTH}, "
            f"{CHANNEL_COUNT}); received {features.shape}."
        )

    labels = np.asarray(labels).reshape(-1)
    if len(features) != len(labels):
        raise ValueError(
            f"{dataset_name} contains {len(features)} feature rows but {len(labels)} labels."
        )
    if not np.issubdtype(labels.dtype, np.integer) or not np.isin(labels, np.arange(1, 11)).all():
        raise ValueError(f"{dataset_name} labels must be integers from 1 through 10.")
    if not np.isfinite(features).all():
        raise ValueError(f"{dataset_name} contains NaN or infinite signal values.")
    return labels.astype(np.int64, copy=False)


def expected_dimension(row_count):
    return f"A1:FAO{row_count + 1}"


def verify_workbook(path, expected_rows):
    """Check the final XLSX container and its one-sheet dimensions."""
    with zipfile.ZipFile(path) as archive:
        invalid_member = archive.testzip()
        if invalid_member:
            raise ValueError(f"Invalid XLSX member: {invalid_member}")

        workbook_xml = archive.read("xl/workbook.xml")
        sheet_count = workbook_xml.count(b"<sheet ")
        if sheet_count != 1:
            raise ValueError(f"Expected one worksheet, found {sheet_count}.")

        with archive.open("xl/worksheets/sheet1.xml") as worksheet_xml:
            start = worksheet_xml.read(8192)
        match = re.search(br'<dimension ref="([A-Z0-9:]+)"', start)
        if not match:
            raise ValueError("Worksheet dimension is missing.")
        actual_dimension = match.group(1).decode("ascii")

    required_dimension = expected_dimension(expected_rows)
    if actual_dimension != required_dimension:
        raise ValueError(
            f"Expected worksheet dimension {required_dimension}, got {actual_dimension}."
        )


def write_workbook(features, class_names, output_path, overwrite):
    """Write one dataset through a temporary XLSX file, then validate and publish it."""
    output_path = Path(output_path)
    temporary_path = output_path.with_name(f"{output_path.stem}.tmp.xlsx")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Use --overwrite to replace it.")
    if temporary_path.exists():
        temporary_path.unlink()

    workbook = xlsxwriter.Workbook(
        temporary_path,
        {"constant_memory": True, "strings_to_numbers": False},
    )
    workbook.use_zip64()
    worksheet = workbook.add_worksheet("Data")
    worksheet.write_row(0, 0, HEADERS)

    for sample_index, (sample, class_name) in enumerate(zip(features, class_names), start=1):
        row = np.empty(EXPECTED_COLUMNS, dtype=object)
        row[0] = class_name
        row[1 : SAMPLE_LENGTH + 1] = sample[:, 0]
        row[SAMPLE_LENGTH + 1 :] = sample[:, 1]
        worksheet.write_row(sample_index, 0, row.tolist())

    workbook.close()
    verify_workbook(temporary_path, len(features))
    os.replace(temporary_path, output_path)
    return output_path.stat().st_size


def main():
    parser = argparse.ArgumentParser(
        description="Export the current 1DCNN demo D train/test splits as XLSX files."
    )
    parser.add_argument(
        "--output-dir",
        default="data/xlsx_1dcnn_demo",
        help="Directory for 1DCNN_train_2048x2.xlsx and 1DCNN_test_2048x2.xlsx.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing final XLSX files after each new workbook validates.",
    )
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent
    pickle_path = project_dir / "data" / "ABCD_Datasets.pickle"
    output_dir = (project_dir / args.output_dir).resolve()
    if not pickle_path.exists():
        raise FileNotFoundError(f"Prepared demo dataset was not found: {pickle_path}")

    with pickle_path.open("rb") as source:
        datasets = pickle.load(source)

    train_features = np.asarray(datasets["train_datasets"]["D"])
    train_labels = validate_dataset(
        train_features, datasets["train_labels"]["D"], "Training dataset"
    )
    train_class_names = [CLASS_NAMES[int(label)] for label in train_labels]
    test_features = np.asarray(datasets["test_datasets"]["D"])
    test_labels = validate_dataset(
        test_features, datasets["test_labels"]["D"], "Test dataset"
    )
    test_class_names = [CLASS_NAMES[int(label)] for label in test_labels]

    if len(train_features) != 19800 or len(test_features) != 750:
        raise ValueError(
            "The prepared demo split does not match the expected 19,800 training and 750 test samples."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = (
        ("Training", train_features, train_class_names, output_dir / "1DCNN_train_2048x2.xlsx"),
        ("Test", test_features, test_class_names, output_dir / "1DCNN_test_2048x2.xlsx"),
    )
    for dataset_name, features, labels, path in outputs:
        print(f"Writing {dataset_name} workbook: {path}")
        file_size = write_workbook(features, labels, path, args.overwrite)
        print(
            f"  verified: {len(features)} rows x {EXPECTED_COLUMNS} columns; "
            f"class_counts={dict(sorted(Counter(labels).items()))}; size={file_size:,} bytes"
        )


if __name__ == "__main__":
    main()
