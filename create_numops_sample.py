# -*- coding: utf-8 -*-
"""Create a balanced, compact XLSX for NumOps classification testing."""

import os
import random
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import scipy.io
import xlsxwriter


OUTPUT_COLUMNS = ["Class_Name", "Load_HP", "DE_time", "FE_time", "RPM"]
SAMPLES_PER_CLASS = 10_000
LOAD_HP = {"A": 0, "B": 1, "C": 2}
DEFAULT_RPM = {"A": 1797.0, "B": 1772.0, "C": 1750.0}
FILE_INFO = {
    "98.mat": (1, "A"), "99.mat": (1, "B"), "100.mat": (1, "C"),
    "119.mat": (2, "A"), "120.mat": (2, "B"), "121.mat": (2, "C"),
    "186.mat": (3, "A"), "187.mat": (3, "B"), "188.mat": (3, "C"),
    "223.mat": (4, "A"), "224.mat": (4, "B"), "225.mat": (4, "C"),
    "106.mat": (5, "A"), "107.mat": (5, "B"), "108.mat": (5, "C"),
    "170.mat": (6, "A"), "171.mat": (6, "B"), "172.mat": (6, "C"),
    "210.mat": (7, "A"), "211.mat": (7, "B"), "212.mat": (7, "C"),
    "131.mat": (8, "A"), "132.mat": (8, "B"), "133.mat": (8, "C"),
    "198.mat": (9, "A"), "199.mat": (9, "B"), "200.mat": (9, "C"),
    "235.mat": (10, "A"), "236.mat": (10, "B"), "237.mat": (10, "C"),
}
CLASS_NAMES = {
    1: "Normal Baseline", 2: "0.007_Inner_Race", 3: "0.014_Ball",
    4: "0.021_Ball", 5: "0.007_Ball", 6: "0.014_Inner_Race",
    7: "0.021_Inner_Race", 8: "0.007_Outer_Race", 9: "0.014_Outer_Race",
    10: "0.021_Outer_Race",
}


def signal(variables, file_number, suffix):
    name = f"X{file_number:03d}_{suffix}"
    if name not in variables:
        raise ValueError(f"{name} was not found.")
    return np.asarray(variables[name]).reshape(-1)


def rpm(variables, file_number, fallback):
    name = f"X{file_number:03d}RPM"
    return float(np.asarray(variables[name]).reshape(-1)[0]) if name in variables else fallback


def main():
    root = Path(__file__).resolve().parent / "1DCNN-master"
    input_dir = root / "data" / "download"
    output_dir = root / "data" / "numops"
    output_path = output_dir / "NumOps_classification_100k.xlsx"
    temporary_path = output_dir / "NumOps_classification_100k.tmp.xlsx"
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f"{output_path} already exists; delete it before recreating.")

    by_class = defaultdict(list)
    for filename, (class_id, load) in FILE_INFO.items():
        by_class[class_id].append((load, input_dir / filename))

    rows = []
    for class_id, sources in sorted(by_class.items()):
        sources.sort()
        counts = [SAMPLES_PER_CLASS // 3] * 3
        counts[0] += SAMPLES_PER_CLASS % 3
        for count, (load, mat_path) in zip(counts, sources):
            variables = {key: value for key, value in scipy.io.loadmat(mat_path).items() if not key.startswith("__")}
            file_number = int(mat_path.stem)
            de_time = signal(variables, file_number, "DE_time")
            fe_time = signal(variables, file_number, "FE_time")
            indices = np.linspace(0, min(len(de_time), len(fe_time)) - 1, count, dtype=int)
            source_rpm = rpm(variables, file_number, DEFAULT_RPM[load])
            rows.extend((CLASS_NAMES[class_id], LOAD_HP[load], float(de_time[i]), float(fe_time[i]), source_rpm) for i in indices)

    if len(rows) != SAMPLES_PER_CLASS * len(CLASS_NAMES):
        raise ValueError(f"Expected 100,000 rows, got {len(rows)}.")
    if Counter(row[0] for row in rows) != Counter({name: SAMPLES_PER_CLASS for name in CLASS_NAMES.values()}):
        raise ValueError("Class balance check failed.")
    random.Random(42).shuffle(rows)

    workbook = xlsxwriter.Workbook(temporary_path, {"constant_memory": True})
    worksheet = workbook.add_worksheet("Data")
    worksheet.write_row(0, 0, OUTPUT_COLUMNS)
    for row_index, row in enumerate(rows, start=1):
        worksheet.write_row(row_index, 0, row)
    workbook.close()

    with zipfile.ZipFile(temporary_path) as archive:
        if archive.testzip() is not None:
            raise ValueError("Generated XLSX is invalid.")
        with archive.open("xl/worksheets/sheet1.xml") as worksheet_xml:
            start = worksheet_xml.read(8192)
    dimension = re.search(br'<dimension ref="([A-Z0-9:]+)"', start)
    if not dimension or dimension.group(1) != b"A1:E100001":
        raise ValueError("Generated XLSX has an unexpected size.")

    os.replace(temporary_path, output_path)
    print(f"Created {output_path}")
    print(f"Rows: {len(rows):,}; per class: {SAMPLES_PER_CLASS:,}; bytes: {output_path.stat().st_size:,}")


if __name__ == "__main__":
    main()
