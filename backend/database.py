from pathlib import Path
import sqlite3
import pandas as pd
from .config import DB_PATH, CSV_PATH, COLUMN_ALIASES, DATE_COL

NORMALIZED_SCHEMA = [
    "date","part_number","serialnumber","typename","stationnumber",
    "stationdescription","failure_code","failure_description","defect",
    "failure_details","action_code","material_code","material_desc","partclass"
]

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    orig_cols = list(df.columns)
    lower = [c.strip().lower() for c in orig_cols]
    mapping = {}
    for canon, aliases in COLUMN_ALIASES.items():
        for idx, col in enumerate(lower):
            if col in aliases:
                mapping[orig_cols[idx]] = canon
    df = df.rename(columns=mapping)
    for c in NORMALIZED_SCHEMA:
        if c not in df.columns:
            df[c] = None
    df = df[NORMALIZED_SCHEMA]
    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    return df

def load_csv_to_df(csv_path: Path = CSV_PATH) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8", engine="python")
    return _normalize_columns(df)

def persist_to_sqlite(df: pd.DataFrame, db_path: Path = DB_PATH, table="failures"):
    con = sqlite3.connect(db_path)
    df.to_sql(table, con, if_exists="replace", index=False)
    con.close()