# -*- coding: utf-8 -*-
"""將 30 個 CWRU MAT 檔轉為固定 10 欄格式的 XLSX。"""

import argparse
import os
import re
import zipfile
from pathlib import Path

import numpy as np
import scipy.io
import xlsxwriter


COLUMNS = [
    "Class_ID", "Class_Name", "Fault_Location", "Fault_Size_Inch", "Load_HP",
    "TimeStep", "DE_time", "FE_time", "BA_time", "RPM",
]

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

CLASS_INFO = {
    1: ("Normal Baseline", "Normal", 0.000),
    2: ("0.007_Inner_Race", "Inner Race", 0.007),
    3: ("0.014_Ball", "Ball", 0.014),
    4: ("0.021_Ball", "Ball", 0.021),
    5: ("0.007_Ball", "Ball", 0.007),
    6: ("0.014_Inner_Race", "Inner Race", 0.014),
    7: ("0.021_Inner_Race", "Inner Race", 0.021),
    8: ("0.007_Outer_Race", "Outer Race", 0.007),
    9: ("0.014_Outer_Race", "Outer Race", 0.014),
    10: ("0.021_Outer_Race", "Outer Race", 0.021),
}

LOAD_HP = {"A": 0, "B": 1, "C": 2}
DEFAULT_RPM = {"A": 1797.0, "B": 1772.0, "C": 1750.0}


def excel_column_name(column_number):
    name = ""
    while column_number:
        column_number, remainder = divmod(column_number - 1, 26)
        name = chr(65 + remainder) + name
    return name


def find_signal(variables, file_number, suffix):
    prefix = f"X{file_number:03d}_{suffix}"
    for name, value in variables.items():
        if name == prefix:
            return np.asarray(value).reshape(-1)
    for name, value in variables.items():
        if name.endswith(f"_{suffix}"):
            return np.asarray(value).reshape(-1)
    return None


def find_rpm(variables, file_number, fallback):
    exact_name = f"X{file_number:03d}RPM"
    if exact_name in variables:
        return float(np.asarray(variables[exact_name]).reshape(-1)[0])
    for name, value in variables.items():
        if name.endswith("RPM"):
            return float(np.asarray(value).reshape(-1)[0])
    return fallback


def verify_xlsx(xlsx_path, row_count):
    with zipfile.ZipFile(xlsx_path) as archive:
        invalid_member = archive.testzip()
        if invalid_member:
            raise ValueError(f"Broken XLSX member: {invalid_member}")
        workbook_xml = archive.read("xl/workbook.xml")
        if workbook_xml.count(b"<sheet ") != 1:
            raise ValueError("The XLSX must contain exactly one worksheet.")
        with archive.open("xl/worksheets/sheet1.xml") as worksheet:
            first_bytes = worksheet.read(8192)
    match = re.search(br'<dimension ref="([A-Z0-9:]+)"', first_bytes)
    expected = f"A1:{excel_column_name(len(COLUMNS))}{row_count + 1}"
    actual = match.group(1).decode("ascii") if match else None
    if actual != expected:
        raise ValueError(f"Expected worksheet size {expected}, got {actual}.")


def convert_file(mat_path, output_dir, overwrite):
    class_id, load_code = FILE_INFO[mat_path.name]
    class_name, fault_location, fault_size = CLASS_INFO[class_id]
    variables = {key: value for key, value in scipy.io.loadmat(mat_path).items() if not key.startswith("__")}
    file_number = int(mat_path.stem)
    de_time = find_signal(variables, file_number, "DE_time")
    fe_time = find_signal(variables, file_number, "FE_time")
    ba_time = find_signal(variables, file_number, "BA_time")
    if de_time is None:
        raise ValueError(f"{mat_path.name}: DE_time was not found.")
    if fe_time is None:
        raise ValueError(f"{mat_path.name}: FE_time was not found.")

    row_count = max(len(de_time), len(fe_time), len(ba_time) if ba_time is not None else 0)
    rpm = find_rpm(variables, file_number, DEFAULT_RPM[load_code])
    output_path = output_dir / f"{mat_path.stem}.xlsx"
    temporary_path = output_dir / f"{mat_path.stem}.tmp.xlsx"
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}")
    if temporary_path.exists():
        temporary_path.unlink()

    workbook = xlsxwriter.Workbook(temporary_path, {"constant_memory": True})
    worksheet = workbook.add_worksheet("Data")
    worksheet.write_row(0, 0, COLUMNS)
    for index in range(row_count):
        de_value = float(de_time[index]) if index < len(de_time) else None
        fe_value = float(fe_time[index]) if index < len(fe_time) else None
        ba_value = float(ba_time[index]) if ba_time is not None and index < len(ba_time) else None
        worksheet.write_row(index + 1, 0, [
            class_id, class_name, fault_location, fault_size, LOAD_HP[load_code],
            index + 1, de_value, fe_value, ba_value, rpm,
        ])
    workbook.close()
    verify_xlsx(temporary_path, row_count)
    os.replace(temporary_path, output_path)
    return row_count


def main():
    parser = argparse.ArgumentParser(description="Convert all CWRU MAT files to the fixed 10-column XLSX schema.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing XLSX files after validation.")
    args = parser.parse_args()
    script_dir = Path(__file__).resolve().parent
    input_dir = script_dir / "1DCNN-master" / "data" / "download"
    output_dir = script_dir / "1DCNN-master" / "data" / "xlsx"
    mat_files = sorted(input_dir.glob("*.mat"))
    if len(mat_files) != 30:
        raise ValueError(f"Expected 30 MAT files in {input_dir}, found {len(mat_files)}.")
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, mat_path in enumerate(mat_files, start=1):
        row_count = convert_file(mat_path, output_dir, args.overwrite)
        print(f"[{index:02d}/30] {mat_path.name}: {row_count} data rows", flush=True)


if __name__ == "__main__":
    main()
