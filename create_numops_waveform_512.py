# -*- coding: utf-8 -*-
"""Create a compact fixed-window waveform classification XLSX for NumOps."""

import os
import random
import re
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import scipy.io
import xlsxwriter


WINDOW_SIZE = 512
WINDOWS_PER_FILE = 100
HEADERS = (
    ["Class_Name"]
    + [f"DE_{point:04d}" for point in range(1, WINDOW_SIZE + 1)]
    + [f"FE_{point:04d}" for point in range(1, WINDOW_SIZE + 1)]
)
FILE_INFO = {
    "98.mat": "Normal Baseline", "99.mat": "Normal Baseline", "100.mat": "Normal Baseline",
    "119.mat": "0.007_Inner_Race", "120.mat": "0.007_Inner_Race", "121.mat": "0.007_Inner_Race",
    "186.mat": "0.014_Ball", "187.mat": "0.014_Ball", "188.mat": "0.014_Ball",
    "223.mat": "0.021_Ball", "224.mat": "0.021_Ball", "225.mat": "0.021_Ball",
    "106.mat": "0.007_Ball", "107.mat": "0.007_Ball", "108.mat": "0.007_Ball",
    "170.mat": "0.014_Inner_Race", "171.mat": "0.014_Inner_Race", "172.mat": "0.014_Inner_Race",
    "210.mat": "0.021_Inner_Race", "211.mat": "0.021_Inner_Race", "212.mat": "0.021_Inner_Race",
    "131.mat": "0.007_Outer_Race", "132.mat": "0.007_Outer_Race", "133.mat": "0.007_Outer_Race",
    "198.mat": "0.014_Outer_Race", "199.mat": "0.014_Outer_Race", "200.mat": "0.014_Outer_Race",
    "235.mat": "0.021_Outer_Race", "236.mat": "0.021_Outer_Race", "237.mat": "0.021_Outer_Race",
}


def excel_column_name(number):
    name = ""
    while number:
        number, remainder = divmod(number - 1, 26)
        name = chr(65 + remainder) + name
    return name


def get_signal(variables, file_number, suffix):
    name = f"X{file_number:03d}_{suffix}"
    if name not in variables:
        raise ValueError(f"{name} was not found.")
    return np.asarray(variables[name]).reshape(-1)


def main():
    project = Path(__file__).resolve().parent / "1DCNN-master"
    input_dir = project / "data" / "download"
    output_dir = project / "data" / "numops"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "NumOps_waveform_512x2_3k.xlsx"
    temporary_path = output_dir / "NumOps_waveform_512x2_3k.tmp.xlsx"
    if output_path.exists():
        raise FileExistsError(f"{output_path} already exists; delete it before recreating.")

    samples = []
    for filename, class_name in sorted(FILE_INFO.items()):
        mat_path = input_dir / filename
        variables = {key: value for key, value in scipy.io.loadmat(mat_path).items() if not key.startswith("__")}
        file_number = int(mat_path.stem)
        de_time = get_signal(variables, file_number, "DE_time")
        fe_time = get_signal(variables, file_number, "FE_time")
        last_start = min(len(de_time), len(fe_time)) - WINDOW_SIZE
        starts = np.linspace(0, last_start, WINDOWS_PER_FILE, dtype=int)
        samples.extend(
            (class_name, de_time[start : start + WINDOW_SIZE], fe_time[start : start + WINDOW_SIZE])
            for start in starts
        )

    expected_rows = len(FILE_INFO) * WINDOWS_PER_FILE
    expected_per_class = WINDOWS_PER_FILE * 3
    if len(samples) != expected_rows or any(len(de) != WINDOW_SIZE or len(fe) != WINDOW_SIZE for _, de, fe in samples):
        raise ValueError("Waveform window generation failed.")
    if set(Counter(label for label, _, _ in samples).values()) != {expected_per_class}:
        raise ValueError("Class balance check failed.")
    random.Random(42).shuffle(samples)

    workbook = xlsxwriter.Workbook(temporary_path, {"constant_memory": True})
    worksheet = workbook.add_worksheet("Data")
    worksheet.write_row(0, 0, HEADERS)
    for row_number, (label, de_window, fe_window) in enumerate(samples, start=1):
        worksheet.write_row(row_number, 0, [label, *de_window, *fe_window])
    workbook.close()

    with zipfile.ZipFile(temporary_path) as archive:
        if archive.testzip() is not None:
            raise ValueError("Generated XLSX is invalid.")
        with archive.open("xl/worksheets/sheet1.xml") as worksheet_xml:
            start = worksheet_xml.read(8192)
    dimension = re.search(br'<dimension ref="([A-Z0-9:]+)"', start)
    expected_dimension = f"A1:{excel_column_name(len(HEADERS))}{expected_rows + 1}".encode("ascii")
    if not dimension or dimension.group(1) != expected_dimension:
        raise ValueError("Generated XLSX has an unexpected size.")

    os.replace(temporary_path, output_path)
    print(f"Created {output_path}")
    print(f"Rows: {expected_rows}; columns: {len(HEADERS)}; per class: {expected_per_class}; bytes: {output_path.stat().st_size:,}")


if __name__ == "__main__":
    main()
