"""Explore POS transaction data (Excel/CSV) before purchase matching."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from configs.camera_timing_config import DATA_DIR, OUTPUTS_REPORTS_DIR

SCHEMA_REPORT_PATH = OUTPUTS_REPORTS_DIR / "pos_schema_report.txt"
COLUMN_SUMMARY_PATH = OUTPUTS_REPORTS_DIR / "pos_column_summary.csv"

POS_EXTENSIONS = (".xlsx", ".xls", ".xlsm", ".csv")
EXCLUDE_DIR_NAMES = {".venv", "venv", "__pycache__", ".git", "node_modules"}

COLUMN_PATTERNS: Dict[str, List[str]] = {
    "Invoice Number": ["invoice_number", "invoice_no", "invoice id", "inv_no", "bill_no"],
    "Bill Number": ["bill_number", "bill_no", "bill id"],
    "Transaction ID": ["transaction_id", "txn_id", "order_id", "sale_id", "receipt_id"],
    "Date": ["order_date", "sale_date", "bill_date", "transaction_date", "date"],
    "Time": ["order_time", "sale_time", "bill_time", "transaction_time", "time"],
    "Timestamp": ["timestamp", "datetime", "transaction_datetime", "sale_datetime"],
    "Product Name": ["product_name", "item_name", "sku_name", "description"],
    "Brand": ["brand_name", "brand"],
    "Category": ["category", "dep_name", "department", "sub_category", "product_category"],
    "Quantity": ["qty", "quantity", "units"],
    "Amount": [
        "total_amount",
        "nmv",
        "net_amount",
        "sale_amount",
        "bill_amount",
        "amount",
        "gmv",
    ],
    "Customer Number": ["customer_number", "customer_phone", "mobile", "phone"],
    "Employee Name": ["salesperson_name", "employee_name", "staff_name", "cashier"],
}


def normalize_col(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(name).lower()).strip()


def find_pos_files(root: Path) -> List[Path]:
    candidates: List[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in POS_EXTENSIONS:
            continue
        if any(part in EXCLUDE_DIR_NAMES for part in path.parts):
            continue
        candidates.append(path)
    return sorted(candidates, key=lambda p: (p.suffix.lower() != ".xlsx", p.name.lower()))


def locate_pos_file(root: Path) -> Path:
    files = find_pos_files(root)
    if not files:
        raise FileNotFoundError(
            f"No POS file found under {root}. "
            f"Expected extensions: {', '.join(POS_EXTENSIONS)}"
        )
    excel_files = [p for p in files if p.suffix.lower() in (".xlsx", ".xls", ".xlsm")]
    chosen = excel_files[0] if excel_files else files[0]
    if len(files) > 1:
        print("Multiple POS files found:")
        for file_path in files:
            marker = " <-- selected" if file_path == chosen else ""
            print(f"  - {file_path.relative_to(root)}{marker}")
    elif chosen.suffix.lower() == ".csv":
        print("Note: No Excel file found; using CSV POS export.")
    return chosen


def load_pos_data(file_path: Path) -> pd.DataFrame:
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(file_path, low_memory=False)
    if suffix == ".xls":
        return pd.read_excel(file_path, engine="xlrd")
    return pd.read_excel(file_path, engine="openpyxl")


def col_matches(norm_col: str, pattern: str) -> bool:
    return norm_col == normalize_col(pattern)


def detect_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    normalized_map = {col: normalize_col(col) for col in df.columns}
    detected: Dict[str, Optional[str]] = {role: None for role in COLUMN_PATTERNS}

    for role, patterns in COLUMN_PATTERNS.items():
        for pattern in patterns:
            for col, norm in normalized_map.items():
                if col_matches(norm, pattern):
                    detected[role] = col
                    break
            if detected[role]:
                break
    return detected


def series_missing_count(series: pd.Series) -> int:
    missing = series.isna()
    if series.dtype == object or pd.api.types.is_string_dtype(series):
        blank = series.astype(str).str.strip().eq("")
        missing = missing | blank
    return int(missing.sum())


def build_column_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for col in df.columns:
        rows.append(
            {
                "column": col,
                "dtype": str(df[col].dtype),
                "missing_count": series_missing_count(df[col]),
                "missing_pct": round(series_missing_count(df[col]) / len(df) * 100, 2)
                if len(df)
                else 0.0,
                "unique_count": int(df[col].nunique(dropna=True)),
                "sample_values": "; ".join(
                    str(value) for value in df[col].dropna().astype(str).unique()[:5]
                ),
            }
        )
    return pd.DataFrame(rows)


def analyze_invoice_structure(
    df: pd.DataFrame, detected: Dict[str, Optional[str]]
) -> Dict[str, Any]:
    invoice_col = detected.get("Invoice Number")
    order_col = detected.get("Transaction ID")
    id_col = invoice_col or order_col

    result: Dict[str, Any] = {
        "invoice_column": invoice_col,
        "order_column": order_col,
        "id_column_used": id_col,
        "duplicate_invoices": 0,
        "invoices_with_multiple_rows": 0,
        "avg_lines_per_invoice": None,
        "max_lines_per_invoice": None,
    }

    if id_col and id_col in df.columns:
        non_null = df[id_col].dropna()
        duplicate_invoices = int(non_null.duplicated().sum())
        result["duplicate_invoices"] = duplicate_invoices
        lines_per_invoice = non_null.value_counts()
        multi = lines_per_invoice[lines_per_invoice > 1]
        result["invoices_with_multiple_rows"] = int(len(multi))
        result["avg_lines_per_invoice"] = round(float(lines_per_invoice.mean()), 2)
        result["max_lines_per_invoice"] = int(lines_per_invoice.max())

    return result


def analyze_datetime_structure(detected: Dict[str, Optional[str]]) -> Dict[str, Any]:
    date_col = detected.get("Date")
    time_col = detected.get("Time")
    ts_col = detected.get("Timestamp")

    separate = date_col is not None and time_col is not None and date_col != time_col
    has_timestamp = ts_col is not None

    return {
        "date_column": date_col,
        "time_column": time_col,
        "timestamp_column": ts_col,
        "date_and_time_separate": separate,
        "timestamp_exists": has_timestamp,
    }


def recommend_matching_columns(
    detected: Dict[str, Optional[str]], datetime_info: Dict[str, Any]
) -> List[str]:
    recommendations: List[str] = []

    if detected.get("Invoice Number"):
        recommendations.append(f"Invoice key: {detected['Invoice Number']}")
    elif detected.get("Transaction ID"):
        recommendations.append(f"Transaction key (fallback): {detected['Transaction ID']}")

    if datetime_info["timestamp_exists"]:
        recommendations.append(f"Timestamp: {datetime_info['timestamp_column']}")
    elif datetime_info["date_and_time_separate"]:
        recommendations.append(
            f"Datetime: combine {datetime_info['date_column']} + {datetime_info['time_column']}"
        )
    elif datetime_info["date_column"]:
        recommendations.append(f"Date only: {datetime_info['date_column']}")

    for role in ("Amount", "Product Name", "Brand", "Category", "Quantity", "Customer Number"):
        if detected.get(role):
            recommendations.append(f"{role}: {detected[role]}")

    return recommendations


def print_sample_unique(df: pd.DataFrame, col: Optional[str], label: str, n: int = 10) -> None:
    print(f"\nSample unique {label} ({n} values):")
    if not col or col not in df.columns:
        print("  (column not detected)")
        return
    values = df[col].dropna().astype(str).unique()[:n]
    for value in values:
        print(f"  - {value}")


def run_exploration() -> None:
    pos_file = locate_pos_file(DATA_DIR)
    df = load_pos_data(pos_file)

    detected = detect_columns(df)
    column_summary = build_column_summary(df)
    datetime_info = analyze_datetime_structure(detected)
    invoice_info = analyze_invoice_structure(df, detected)

    duplicate_rows = int(df.duplicated().sum())
    missing_timestamps = 0
    if datetime_info["timestamp_exists"] and datetime_info["timestamp_column"]:
        missing_timestamps = series_missing_count(df[datetime_info["timestamp_column"]])
    elif datetime_info["date_column"]:
        date_missing = series_missing_count(df[datetime_info["date_column"]])
        time_missing = 0
        if datetime_info["time_column"]:
            time_missing = series_missing_count(df[datetime_info["time_column"]])
        missing_timestamps = max(date_missing, time_missing)

    missing_invoice_ids = 0
    id_col = invoice_info["id_column_used"]
    if id_col:
        missing_invoice_ids = series_missing_count(df[id_col])

    recommendations = recommend_matching_columns(detected, datetime_info)

    report_lines: List[str] = []
    append = report_lines.append

    append("POS DATA EXPLORATION REPORT")
    append("=" * 60)
    append(f"File: {pos_file.name}")
    append(f"Path: {pos_file}")
    append(f"Rows: {len(df)}")
    append(f"Columns: {len(df.columns)}")
    append("")
    append("Column names:")
    for col in df.columns:
        append(f"  - {col}")
    append("")
    append("Data types:")
    for col, dtype in df.dtypes.items():
        append(f"  {col}: {dtype}")
    append("")
    append("Missing value counts:")
    for col in df.columns:
        append(f"  {col}: {series_missing_count(df[col])}")
    append("")
    append("Detected column roles:")
    for role, col in detected.items():
        append(f"  {role}: {col or 'NOT FOUND'}")
    append("")
    append("Data quality:")
    append(f"  Missing timestamps: {missing_timestamps}")
    append(f"  Missing invoice/transaction IDs: {missing_invoice_ids}")
    append(f"  Duplicate invoice IDs (extra rows): {invoice_info['duplicate_invoices']}")
    append(f"  Duplicate full rows: {duplicate_rows}")
    append(f"  Invoices with multiple product lines: {invoice_info['invoices_with_multiple_rows']}")
    if invoice_info["avg_lines_per_invoice"] is not None:
        append(f"  Avg lines per invoice: {invoice_info['avg_lines_per_invoice']}")
        append(f"  Max lines per invoice: {invoice_info['max_lines_per_invoice']}")
    append("")
    append("Datetime structure:")
    append(f"  Date and time separate: {datetime_info['date_and_time_separate']}")
    append(f"  Timestamp column exists: {datetime_info['timestamp_exists']}")
    append(f"  Date column: {datetime_info['date_column']}")
    append(f"  Time column: {datetime_info['time_column']}")
    append(f"  Timestamp column: {datetime_info['timestamp_column']}")
    append("")
    append("Columns suitable for purchase matching:")
    for line in recommendations:
        append(f"  - {line}")

    SCHEMA_REPORT_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    column_summary.to_csv(COLUMN_SUMMARY_PATH, index=False)

    # Console output
    print("=" * 60)
    print("POS DATA EXPLORATION")
    print("=" * 60)
    print(f"File name: {pos_file.name}")
    print(f"Full path: {pos_file}")
    print(f"Number of rows: {len(df)}")
    print(f"Number of columns: {len(df.columns)}")
    print("\nColumn names:")
    for col in df.columns:
        print(f"  {col}")
    print("\nData types:")
    print(df.dtypes.to_string())
    print("\nMissing value counts:")
    for col in df.columns:
        print(f"  {col}: {series_missing_count(df[col])}")

    print("\n" + "=" * 60)
    print("FIRST 20 ROWS")
    print("=" * 60)
    print(df.head(20).to_string())

    print("\n" + "=" * 60)
    print("LAST 20 ROWS")
    print("=" * 60)
    print(df.tail(20).to_string())

    print("\n" + "=" * 60)
    print("DETECTED COLUMN ROLES")
    print("=" * 60)
    for role, col in detected.items():
        print(f"  {role}: {col or 'NOT FOUND'}")

    print("\n" + "=" * 60)
    print("DATA QUALITY REPORT")
    print("=" * 60)
    print(f"Missing timestamps: {missing_timestamps}")
    print(f"Missing invoice IDs: {missing_invoice_ids}")
    print(f"Duplicate invoice IDs (repeated on multiple rows): {invoice_info['duplicate_invoices']}")
    print(f"Duplicate full rows: {duplicate_rows}")
    print(f"Invoices with multiple product lines: {invoice_info['invoices_with_multiple_rows']}")
    if invoice_info["avg_lines_per_invoice"] is not None:
        print(f"Avg product lines per invoice: {invoice_info['avg_lines_per_invoice']}")
        print(f"Max product lines per invoice: {invoice_info['max_lines_per_invoice']}")

    print("\n" + "=" * 60)
    print("DATETIME STRUCTURE")
    print("=" * 60)
    print(f"Date and time are separate columns: {datetime_info['date_and_time_separate']}")
    print(f"Dedicated timestamp column exists: {datetime_info['timestamp_exists']}")
    print(f"Multiple products per invoice: {invoice_info['invoices_with_multiple_rows'] > 0}")

    print_sample_unique(df, detected.get("Brand"), "Brand")
    print_sample_unique(df, detected.get("Product Name"), "Product")
    print_sample_unique(df, detected.get("Category"), "Category")

    print("\n" + "=" * 60)
    print("COLUMNS SUITABLE FOR PURCHASE MATCHING")
    print("=" * 60)
    for line in recommendations:
        print(f"  - {line}")

    print("\n" + "=" * 60)
    print("OUTPUT FILES")
    print("=" * 60)
    print(f"  {SCHEMA_REPORT_PATH}")
    print(f"  {COLUMN_SUMMARY_PATH}")


if __name__ == "__main__":
    try:
        run_exploration()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
