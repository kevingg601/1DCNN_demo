# -*- coding: utf-8 -*-
"""Create a long-format time-series classification XLSX for NumOps."""

import os
import random
import re
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import scipy.io
import xlsxwriter


WINDOW_SIZE = 512
SEGMENTS_PER_CLASS = 100
HEADERS = ["Series_ID", "Timestamp", "DE_time", "FE_time", "Class_Name"]
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


def signal(variables, file_number, suffix):
    name = f"X{file_number:03d}_{suffix}"
    if name not in variables:
        raise ValueError(f"{name} was not found.")
    return np.asarray(variables[name]).reshape(-1)


def main():
    project = Path(__file__).resolve().parent / "1DCNN-master"
    input_dir = project / "data" / "download"
    output_dir = project / "data" / "numops"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "NumOps_timeseries_512x2_1k.xlsx"
    temporary_path = output_dir / "NumOps_timeseries_512x2_1k.tmp.xlsx"
    if output_path.exists():
        raise FileExistsError(f"{output_path} already exists; delete it before recreating.")

    by_class = defaultdict(list)
    for filename, class_name in FILE_INFO.items():
        by_class[class_name].append(input_dir / filename)

    segments = []
    for class_name, paths in sorted(by_class.items()):
        paths.sort()
        counts = [SEGMENTS_PER_CLASS // 3] * 3
        counts[0] += SEGMENTS_PER_CLASS % 3
        for segment_count, mat_path in zip(counts, paths):
            variables = {key: value for key, value in scipy.io.loadmat(mat_path).items() if not key.startswith("__")}
            file_number = int(mat_path.stem)
            de_time = signal(variables, file_number, "DE_time")
            fe_time = signal(variables, file_number, "FE_time")
            starts = np.linspace(0, min(len(de_time), len(fe_time)) - WINDOW_SIZE, segment_count, dtype=int)
            for ordinal, start in enumerate(starts, start=1):
                series_id = f"{mat_path.stem}_segment_{ordinal:03d}"
                segments.append((series_id, class_name, de_time[start : start + WINDOW_SIZE], fe_time[start : start + WINDOW_SIZE]))

    if len(segments) != SEGMENTS_PER_CLASS * len(by_class):
        raise ValueError("Segment count check failed.")
    if set(Counter(label for _, label, _, _ in segments).values()) != {SEGMENTS_PER_CLASS}:
        raise ValueError("Class balance check failed.")
    random.Random(42).shuffle(segments)

    workbook = xlsxwriter.Workbook(temporary_path, {"constant_memory": True})
    worksheet = workbook.add_worksheet("Data")
    timestamp_format = workbook.add_format({"num_format": "yyyy-mm-dd hh:mm:ss.000"})
    worksheet.write_row(0, 0, HEADERS)
    timestamp_start = datetime(2026, 1, 1)
    row_number = 1
    for series_id, class_name, de_window, fe_window in segments:
        for time_index, (de_value, fe_value) in enumerate(zip(de_window, fe_window)):
            worksheet.write(row_number, 0, series_id)
            worksheet.write_datetime(row_number, 1, timestamp_start + timedelta(milliseconds=time_index), timestamp_format)
            worksheet.write_number(row_number, 2, float(de_value))
            worksheet.write_number(row_number, 3, float(fe_value))
            worksheet.write(row_number, 4, class_name)
            row_number += 1
    workbook.close()

    expected_rows = len(segments) * WINDOW_SIZE
    with zipfile.ZipFile(temporary_path) as archive:
        if archive.testzip() is not None:
            raise ValueError("Generated XLSX is invalid.")
        with archive.open("xl/worksheets/sheet1.xml") as worksheet_xml:
            start = worksheet_xml.read(8192)
    dimension = re.search(br'<dimension ref="([A-Z0-9:]+)"', start)
    if not dimension or dimension.group(1) != f"A1:E{expected_rows + 1}".encode("ascii"):
        raise ValueError("Generated XLSX has an unexpected size.")

    os.replace(temporary_path, output_path)
    print(f"Created {output_path}")
    print(f"Segments: {len(segments)}; rows: {expected_rows}; per class: {SEGMENTS_PER_CLASS}; bytes: {output_path.stat().st_size:,}")


if __name__ == "__main__":
    main()
