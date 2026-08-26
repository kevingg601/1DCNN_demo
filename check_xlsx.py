import glob
import pandas as pd

files = sorted(glob.glob(r'd:\PYTHON\1DCNN-master\1DCNN-master\data\xlsx\*.xlsx'))
print(f"Total files in xlsx folder: {len(files)}")

ok_count = 0
for f in files:
    try:
        df = pd.read_excel(f, nrows=1, engine='openpyxl')
        cols = list(df.columns)
        if len(cols) == 9:
            ok_count += 1
        else:
            print(f"File {f} has {len(cols)} columns: {cols}")
    except Exception as e:
        print(f"File {f} error: {e}")

print(f"Fully aligned 9-column XLSX count: {ok_count} / 30")
