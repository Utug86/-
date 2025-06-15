import pandas as pd
from pathlib import Path
from rag_file_utils import clean_html_from_cell
import logging

logger = logging.getLogger("rag_table_utils")
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] [rag_table_utils] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

def process_table_for_rag(
    file_path: Path,
    columns=None,
    filter_expr=None,
    add_headers=True,
    row_delim="\n"
) -> str:
    ext = file_path.suffix.lower()
    try:
        logger.info(f"Processing table for RAG: {file_path.name}")
        if ext == ".csv":
            df = pd.read_csv(file_path, usecols=columns)
        elif ext in [".xlsx", ".xls", ".xlsm"]:
            df = pd.read_excel(file_path, usecols=columns)
        else:
            logger.error("Not a table file")
            raise ValueError("Not a table file")
        if filter_expr:
            df = df.query(filter_expr)
        for col in df.columns:
            df[col] = df[col].apply(clean_html_from_cell)
        rows = []
        colnames = list(df.columns)
        for idx, row in df.iterrows():
            row_items = [f"{col}: {row[col]}" for col in colnames]
            rows.append(" | ".join(row_items))
        result = ""
        if add_headers:
            header = " | ".join(colnames)
            result = header + row_delim
        result += row_delim.join(rows)
        logger.info(f"Table processed for RAG: {file_path.name}, rows: {len(df)}")
        return result
    except Exception as e:
        logger.error(f"process_table_for_rag error: {e}")
        return f"[Ошибка обработки таблицы для RAG]: {e}"
