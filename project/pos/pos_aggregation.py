"""Aggregate line-item POS data to invoice-level transactions."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from configs.camera_timing_config import OUTPUTS_DIR, POS_CSV_PATH

INPUT_PATH = POS_CSV_PATH
OUTPUT_CSV = OUTPUTS_DIR / "aggregated_transactions.csv"
OUTPUT_JSON = OUTPUTS_DIR / "aggregated_transactions.json"


def load_pos_lines(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"POS file not found: {path}")
    return pd.read_csv(path, low_memory=False)


def build_transaction_datetime(df: pd.DataFrame) -> pd.Series:
    combined = (
        df["order_date"].astype(str).str.strip()
        + " "
        + df["order_time"].astype(str).str.strip()
    )
    return pd.to_datetime(combined, format="%d-%m-%Y %H:%M:%S", errors="coerce")


def unique_sorted(values: pd.Series) -> List[str]:
    cleaned = (
        values.dropna()
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
        .dropna()
        .unique()
        .tolist()
    )
    return sorted(cleaned)


def aggregate_invoices(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["transaction_datetime"] = build_transaction_datetime(work)

    grouped = work.groupby("invoice_number", sort=True)

    rows: List[Dict[str, Any]] = []
    for invoice_number, group in grouped:
        rows.append(
            {
                "invoice_number": invoice_number,
                "order_id": int(group["order_id"].iloc[0]),
                "transaction_datetime": group["transaction_datetime"].iloc[0],
                "customer_number": int(group["customer_number"].iloc[0]),
                "salesperson_name": str(group["salesperson_name"].iloc[0]).strip(),
                "product_names": unique_sorted(group["product_name"]),
                "brand_names": unique_sorted(group["brand_name"]),
                "categories": unique_sorted(group["dep_name"]),
                "total_quantity": int(group["qty"].sum()),
                "total_amount": round(float(group["total_amount"].sum()), 2),
            }
        )

    aggregated = pd.DataFrame(rows)
    aggregated["transaction_datetime"] = pd.to_datetime(aggregated["transaction_datetime"])
    aggregated = aggregated.sort_values("transaction_datetime").reset_index(drop=True)
    return aggregated


def aggregated_to_json_records(aggregated: pd.DataFrame) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for row in aggregated.to_dict(orient="records"):
        ts = row["transaction_datetime"]
        row["transaction_datetime"] = (
            ts.isoformat(sep=" ") if hasattr(ts, "isoformat") else str(ts)
        )
        records.append(row)
    return records


def aggregated_to_csv(aggregated: pd.DataFrame) -> pd.DataFrame:
    export = aggregated.copy()
    export["transaction_datetime"] = export["transaction_datetime"].dt.strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    for col in ("product_names", "brand_names", "categories"):
        export[col] = export[col].apply(json.dumps, ensure_ascii=False)
    return export


def print_analysis(line_items: pd.DataFrame, aggregated: pd.DataFrame) -> None:
    print("=" * 60)
    print("POS INVOICE AGGREGATION ANALYSIS")
    print("=" * 60)
    print(f"Line items loaded: {len(line_items)}")
    print(f"Total invoices: {len(aggregated)}")
    print(f"Average basket value: {aggregated['total_amount'].mean():.2f}")
    print(f"Highest invoice amount: {aggregated['total_amount'].max():.2f}")

    brand_counts = Counter(line_items["brand_name"].dropna().astype(str).str.strip())
    category_counts = Counter(line_items["dep_name"].dropna().astype(str).str.strip())

    print("\nTop brands sold (by line items):")
    for brand, count in brand_counts.most_common(10):
        print(f"  {brand}: {count}")

    print("\nTop categories sold (by line items):")
    for category, count in category_counts.most_common(10):
        print(f"  {category}: {count}")

    print("\nTop brands by quantity:")
    brand_qty = line_items.groupby("brand_name")["qty"].sum().sort_values(ascending=False)
    for brand, qty in brand_qty.head(10).items():
        print(f"  {brand}: {int(qty)}")

    print("\nSample invoice (first row):")
    sample = aggregated.iloc[0]
    print(f"  invoice_number: {sample['invoice_number']}")
    print(f"  transaction_datetime: {sample['transaction_datetime']}")
    print(f"  total_amount: {sample['total_amount']}")
    print(f"  brand_names: {sample['brand_names'][:5]}")


def main() -> None:
    print(f"Loading: {INPUT_PATH.name}")
    line_items = load_pos_lines(INPUT_PATH)
    aggregated = aggregate_invoices(line_items)

    csv_export = aggregated_to_csv(aggregated)
    csv_export.to_csv(OUTPUT_CSV, index=False)

    json_records = aggregated_to_json_records(aggregated)
    OUTPUT_JSON.write_text(
        json.dumps(json_records, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print_analysis(line_items, aggregated)

    print("\n" + "=" * 60)
    print("OUTPUT FILES")
    print("=" * 60)
    print(f"  {OUTPUT_CSV}")
    print(f"  {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
