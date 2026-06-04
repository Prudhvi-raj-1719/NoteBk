"""Retail intelligence from normalized CCTV events, POS transactions, and purchase matches."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

from configs.camera_timing_config import OUTPUTS_CHARTS_DIR, OUTPUTS_DIR, PROJECT_ROOT
from events.event_time import event_datetime_for_matching

NORMALIZED_EVENT_FILES: Dict[str, Path] = {
    "CAM1": OUTPUTS_DIR / "cam1_events_normalized.jsonl",
    "CAM2": OUTPUTS_DIR / "cam2_events_normalized.jsonl",
    "CAM5": OUTPUTS_DIR / "cam5_events_normalized.jsonl",
}
TRANSACTIONS_PATH = OUTPUTS_DIR / "aggregated_transactions.json"
PURCHASE_MATCHES_PATH = OUTPUTS_DIR / "purchase_matches.json"
SUMMARY_PATH = OUTPUTS_DIR / "analytics_summary.json"
REPORT_PATH = OUTPUTS_DIR / "analytics_report.md"
CHARTS_DIR = OUTPUTS_CHARTS_DIR

ZONE_ENTER = "ZONE_ENTER"
DWELL_COMPLETED = "DWELL_COMPLETED"
ZONE_DWELL = "ZONE_DWELL"
CAM5_NON_BRAND_ZONES = frozenset({"PaymentArea", "BillingQueue", "OUT_OF_ZONE"})


@dataclass
class AnalyticsBundle:
    cctv: Dict[str, Any] = field(default_factory=dict)
    pos: Dict[str, Any] = field(default_factory=dict)
    matching: Dict[str, Any] = field(default_factory=dict)
    combined: Dict[str, Any] = field(default_factory=dict)
    executive_insights: List[str] = field(default_factory=list)
    chart_paths: List[str] = field(default_factory=list)


def normalize_brand_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    events: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                events.append(json.loads(stripped))
    return events


def load_all_cctv_events() -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    events: List[Dict[str, Any]] = []
    source_counts: Dict[str, int] = {}
    for camera_id, path in NORMALIZED_EVENT_FILES.items():
        camera_events = load_jsonl(path)
        source_counts[camera_id] = len(camera_events)
        for event in camera_events:
            event.setdefault("camera", camera_id)
            events.append(event)
    events.sort(
        key=lambda item: event_datetime_for_matching(item) or item.get("timestamp", "")
    )
    return events, source_counts


def load_json_array(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in {path}")
    return data


def event_zone_name(event: Dict[str, Any]) -> Optional[str]:
    meta = event.get("metadata") or {}
    return event.get("zone") or meta.get("sku_zone")


def is_brand_zone_event(event: Dict[str, Any]) -> bool:
    zone = event_zone_name(event)
    if not zone or zone in CAM5_NON_BRAND_ZONES:
        return False
    return event.get("event_type") == ZONE_ENTER


def compute_cctv_analytics(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    brand_visit_counts: Counter[str] = Counter()
    dwell_totals: Dict[str, float] = defaultdict(float)
    dwell_samples: Dict[str, int] = defaultdict(int)

    journeys: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

    for event in events:
        visitor_id = str(event.get("visitor_id", "unknown"))
        event_dt = event_datetime_for_matching(event) or ""
        zone = event_zone_name(event)

        if is_brand_zone_event(event) and zone:
            brand_visit_counts[str(zone)] += 1
            journeys[visitor_id].append((event_dt, str(zone)))

        if event.get("event_type") in (DWELL_COMPLETED, ZONE_DWELL) and zone:
            dwell = event.get("dwell_seconds")
            if dwell is None and event.get("dwell_ms") is not None:
                dwell = float(event["dwell_ms"]) / 1000.0
            if isinstance(dwell, (int, float)):
                dwell_totals[str(zone)] += float(dwell)
                dwell_samples[str(zone)] += 1

    zone_dwell_times = {
        zone: {
            "total_dwell_seconds": round(total, 1),
            "average_dwell_seconds": round(total / dwell_samples[zone], 1)
            if dwell_samples[zone]
            else 0.0,
            "dwell_event_count": dwell_samples[zone],
        }
        for zone, total in sorted(dwell_totals.items(), key=lambda item: -item[1])
    }

    sorted_visits = brand_visit_counts.most_common()
    most_visited = sorted_visits[:5] if sorted_visits else []
    least_visited = (
        sorted(brand_visit_counts.items(), key=lambda item: (item[1], item[0]))[:5]
        if brand_visit_counts
        else []
    )

    top_journeys: List[Dict[str, Any]] = []
    for visitor_id, steps in journeys.items():
        steps_sorted = sorted(steps, key=lambda item: item[0])
        zones = [zone for _, zone in steps_sorted]
        if not zones:
            continue
        top_journeys.append(
            {
                "visitor_id": visitor_id,
                "zone_sequence": zones,
                "unique_zones": len(set(zones)),
                "total_zone_entries": len(zones),
            }
        )

    top_journeys.sort(
        key=lambda item: (item["total_zone_entries"], item["unique_zones"]),
        reverse=True,
    )

    return {
        "total_events": len(events),
        "brand_visit_counts": dict(sorted_visits),
        "zone_dwell_times": zone_dwell_times,
        "most_visited_brands": [
            {"brand": brand, "visits": count} for brand, count in most_visited
        ],
        "least_visited_brands": [
            {"brand": brand, "visits": count} for brand, count in least_visited
        ],
        "top_5_visitor_journeys": top_journeys[:5],
    }


def compute_pos_analytics(transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_revenue = 0.0
    revenue_by_brand: Dict[str, float] = defaultdict(float)
    revenue_by_category: Dict[str, float] = defaultdict(float)
    product_counter: Counter[str] = Counter()

    for txn in transactions:
        amount = float(txn.get("total_amount") or 0.0)
        total_revenue += amount

        brands = txn.get("brand_names") or []
        categories = txn.get("categories") or []
        products = txn.get("product_names") or []

        if brands:
            share = amount / len(brands)
            for brand in brands:
                revenue_by_brand[str(brand)] += share
        if categories:
            share = amount / len(categories)
            for category in categories:
                revenue_by_category[str(category)] += share

        for product in products:
            product_counter[str(product)] += 1

    total_invoices = len(transactions)
    average_basket = total_revenue / total_invoices if total_invoices else 0.0

    top_products = [
        {"product": name, "invoice_line_count": count}
        for name, count in product_counter.most_common(10)
    ]

    return {
        "total_invoices": total_invoices,
        "total_revenue": round(total_revenue, 2),
        "average_basket_value": round(average_basket, 2),
        "revenue_by_brand": dict(
            sorted(revenue_by_brand.items(), key=lambda item: -item[1])
        ),
        "revenue_by_category": dict(
            sorted(revenue_by_category.items(), key=lambda item: -item[1])
        ),
        "top_selling_products": top_products,
    }


def compute_matching_analytics(
    matches: List[Dict[str, Any]],
) -> Dict[str, Any]:
    matched = [row for row in matches if row.get("match_found")]
    unmatched = [row for row in matches if not row.get("match_found")]
    scores = [float(row.get("confidence_score", 0.0)) for row in matched]

    conversion_opportunities: List[Dict[str, Any]] = []
    for row in unmatched:
        journey = row.get("candidate_customer_journey") or {}
        all_events = (
            journey.get("cam1_events", [])
            + journey.get("cam2_events", [])
            + journey.get("cam5_events", [])
        )
        zones = [
            z
            for event in all_events
            if event.get("event_type") == ZONE_ENTER
            for z in [event_zone_name(event)]
            if z
        ]
        conversion_opportunities.append(
            {
                "invoice_number": row.get("invoice_number"),
                "transaction_datetime": row.get("transaction_datetime"),
                "brands_purchased": row.get("brands_purchased", []),
                "reason": row.get("reason", "unmatched"),
                "nearby_cctv_event_count": len(all_events),
                "zones_in_window": zones,
            }
        )

    return {
        "matched_invoices": len(matched),
        "unmatched_invoices": len(unmatched),
        "average_confidence": round(sum(scores) / len(scores), 2) if scores else None,
        "conversion_opportunities": conversion_opportunities,
        "matched_invoice_numbers": [
            row.get("invoice_number") for row in matched
        ],
        "unmatched_invoice_numbers": [
            row.get("invoice_number") for row in unmatched
        ],
    }


def map_cctv_to_pos_brands(
    cctv_brands: List[str],
    pos_brands: List[str],
) -> Dict[str, Optional[str]]:
    pos_by_key = {normalize_brand_key(b): b for b in pos_brands}
    mapping: Dict[str, Optional[str]] = {}
    for cctv in cctv_brands:
        mapping[cctv] = pos_by_key.get(normalize_brand_key(cctv))
    return mapping


def compute_combined_insights(
    cctv: Dict[str, Any],
    pos: Dict[str, Any],
    matching: Dict[str, Any],
) -> Dict[str, Any]:
    visit_counts: Dict[str, int] = cctv.get("brand_visit_counts", {})
    revenue_by_brand: Dict[str, float] = pos.get("revenue_by_brand", {})

    cctv_brands = list(visit_counts.keys())
    pos_brands = list(revenue_by_brand.keys())
    brand_mapping = map_cctv_to_pos_brands(cctv_brands, pos_brands)

    visits_by_pos_key: Dict[str, int] = defaultdict(int)
    for cctv_zone, count in visit_counts.items():
        mapped = brand_mapping.get(cctv_zone)
        key = normalize_brand_key(mapped) if mapped else normalize_brand_key(cctv_zone)
        visits_by_pos_key[key] += count

    revenue_by_key = {
        normalize_brand_key(brand): (brand, revenue)
        for brand, revenue in revenue_by_brand.items()
    }

    all_keys = set(visits_by_pos_key) | set(revenue_by_key)

    comparison: List[Dict[str, Any]] = []
    for key in all_keys:
        visits = visits_by_pos_key.get(key, 0)
        pos_name, revenue = revenue_by_key.get(key, ("", 0.0))
        comparison.append(
            {
                "brand_key": key,
                "pos_brand_name": pos_name or key,
                "cctv_visits": visits,
                "revenue": round(revenue, 2),
            }
        )

    comparison.sort(key=lambda item: (-item["cctv_visits"], -item["revenue"]))

    max_visits = max((item["cctv_visits"] for item in comparison), default=0)
    max_revenue = max((item["revenue"] for item in comparison), default=0.0)

    high_interest_low_sales = [
        item
        for item in comparison
        if item["cctv_visits"] >= max(1, max_visits * 0.5) and item["revenue"] == 0
    ]
    high_sales_low_engagement = [
        item
        for item in comparison
        if item["revenue"] >= max_revenue * 0.25
        and item["cctv_visits"] <= max(1, max_visits * 0.25)
    ]

    most_engaged = max(comparison, key=lambda item: item["cctv_visits"], default=None)
    most_purchased = max(comparison, key=lambda item: item["revenue"], default=None)

    observations: List[str] = []
    if high_interest_low_sales:
        names = ", ".join(item["pos_brand_name"] for item in high_interest_low_sales[:3])
        observations.append(f"High interest, low sales: {names}.")
    if high_sales_low_engagement:
        names = ", ".join(item["pos_brand_name"] for item in high_sales_low_engagement[:3])
        observations.append(f"High sales, low engagement (CCTV window): {names}.")
    if most_engaged:
        observations.append(
            f"Most engaged zone (visits): {most_engaged['pos_brand_name']} "
            f"({most_engaged['cctv_visits']} entries)."
        )
    if most_purchased:
        observations.append(
            f"Most purchased brand (revenue): {most_purchased['pos_brand_name']} "
            f"(INR {most_purchased['revenue']:,.2f})."
        )
    if matching.get("matched_invoices", 0) == 0:
        observations.append(
            "No invoices matched CCTV events in the purchase-matching window; "
            "combined visit vs purchase signals use full-day POS vs clip-period CCTV."
        )

    return {
        "brands_visited_cctv": visit_counts,
        "brands_purchased_pos": revenue_by_brand,
        "brand_name_mapping_cctv_to_pos": brand_mapping,
        "brand_comparison": comparison,
        "high_interest_low_sales": high_interest_low_sales,
        "high_sales_low_engagement": high_sales_low_engagement,
        "most_engaged_zone": most_engaged,
        "most_purchased_brand": most_purchased,
        "observations": observations,
    }


def generate_executive_insights(bundle: AnalyticsBundle) -> List[str]:
    cctv = bundle.cctv
    pos = bundle.pos
    matching = bundle.matching
    combined = bundle.combined

    insights: List[str] = []

    if cctv.get("most_visited_brands"):
        top = cctv["most_visited_brands"][0]
        insights.append(
            f"{top['brand']} recorded the highest shelf traffic with "
            f"{top['visits']} zone entries in the analyzed CCTV clips."
        )

    if cctv.get("zone_dwell_times"):
        top_dwell_zone = max(
            cctv["zone_dwell_times"].items(),
            key=lambda item: item[1]["total_dwell_seconds"],
        )
        insights.append(
            f"{top_dwell_zone[0]} accumulated the longest total dwell time "
            f"({top_dwell_zone[1]['total_dwell_seconds']:.1f}s) across visitors."
        )

    if pos.get("revenue_by_brand"):
        top_brand, top_rev = next(iter(pos["revenue_by_brand"].items()))
        insights.append(
            f"{top_brand} generated the highest revenue (INR {top_rev:,.2f}) "
            f"across {pos['total_invoices']} invoices."
        )

    if pos.get("revenue_by_category"):
        top_cat, cat_rev = next(iter(pos["revenue_by_category"].items()))
        insights.append(
            f"Category '{top_cat}' led sales with INR {cat_rev:,.2f} in attributed revenue."
        )

    insights.append(
        f"Average basket value was INR {pos.get('average_basket_value', 0):,.2f} "
        f"on total revenue of INR {pos.get('total_revenue', 0):,.2f}."
    )

    matched = matching.get("matched_invoices", 0)
    unmatched = matching.get("unmatched_invoices", 0)
    insights.append(
        f"Purchase matching linked {matched} of {matched + unmatched} invoices to "
        f"CCTV activity (average confidence: "
        f"{matching.get('average_confidence') or 'N/A'})."
    )

    for observation in combined.get("observations", [])[:3]:
        insights.append(observation)

    high_interest = combined.get("high_interest_low_sales", [])
    if high_interest:
        brand = high_interest[0].get("pos_brand_name", "A brand")
        insights.append(
            f"{brand} attracted strong visitor attention but shows no attributed "
            f"purchase revenue in the POS extract - review conversion levers."
        )

    high_sales_low = combined.get("high_sales_low_engagement", [])
    if high_sales_low:
        brand = high_sales_low[0].get("pos_brand_name", "A brand")
        insights.append(
            f"{brand} sold well in POS but had limited visible engagement in the "
            f"CCTV analysis window—demand may occur outside recorded clips."
        )

    if pos.get("top_selling_products"):
        product = pos["top_selling_products"][0]["product"]
        short_name = product[:80] + ("..." if len(product) > 80 else "")
        insights.append(f"Top product by appearance on invoices: {short_name}.")

    journeys = cctv.get("top_5_visitor_journeys", [])
    if journeys:
        journey = journeys[0]
        seq = " → ".join(journey["zone_sequence"][:6])
        if len(journey["zone_sequence"]) > 6:
            seq += " → ..."
        insights.append(
            f"Visitor {journey['visitor_id']} had the richest journey ({seq})."
        )

    return insights[:10]


def save_bar_chart(
    title: str,
    labels: List[str],
    values: List[float],
    ylabel: str,
    filename: str,
    bundle: AnalyticsBundle,
    top_n: int = 12,
) -> None:
    if not values:
        return

    pairs = sorted(zip(labels, values), key=lambda item: -item[1])[:top_n]
    chart_labels = [label for label, _ in pairs]
    chart_values = [value for _, value in pairs]

    fig, ax = plt.subplots(figsize=(10, max(4, len(chart_labels) * 0.35)))
    ax.barh(chart_labels[::-1], chart_values[::-1], color="#3b82f6")
    ax.set_title(title)
    ax.set_xlabel(ylabel)
    fig.tight_layout()

    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    path = CHARTS_DIR / filename
    fig.savefig(path, dpi=120)
    plt.close(fig)
    bundle.chart_paths.append(str(path.relative_to(PROJECT_ROOT)))


def generate_charts(bundle: AnalyticsBundle) -> None:
    cctv = bundle.cctv
    pos = bundle.pos

    save_bar_chart(
        "Brand Visits (ZONE_ENTER)",
        list(cctv.get("brand_visit_counts", {}).keys()),
        [float(v) for v in cctv.get("brand_visit_counts", {}).values()],
        "Visit count",
        "brand_visits.png",
        bundle,
    )

    dwell = cctv.get("zone_dwell_times", {})
    save_bar_chart(
        "Total Zone Dwell Time",
        list(dwell.keys()),
        [info["total_dwell_seconds"] for info in dwell.values()],
        "Seconds",
        "dwell_times.png",
        bundle,
    )

    save_bar_chart(
        "Revenue by Brand",
        list(pos.get("revenue_by_brand", {}).keys()),
        [float(v) for v in pos.get("revenue_by_brand", {}).values()],
        "Revenue (INR)",
        "revenue_by_brand.png",
        bundle,
    )

    save_bar_chart(
        "Revenue by Category",
        list(pos.get("revenue_by_category", {}).keys()),
        [float(v) for v in pos.get("revenue_by_category", {}).values()],
        "Revenue (INR)",
        "revenue_by_category.png",
        bundle,
    )


def build_markdown_report(bundle: AnalyticsBundle) -> str:
    lines = [
        "# Retail Analytics Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Executive Summary",
        "",
    ]
    for idx, insight in enumerate(bundle.executive_insights, start=1):
        lines.append(f"{idx}. {insight}")

    lines.extend(["", "## CCTV Analytics", ""])
    cctv = bundle.cctv
    lines.append(f"- Total events analyzed: **{cctv.get('total_events', 0)}**")
    lines.append("")
    lines.append("### Brand visit counts")
    lines.append("")
    for brand, count in cctv.get("brand_visit_counts", {}).items():
        lines.append(f"- {brand}: {count}")
    lines.append("")
    lines.append("### Most visited brands")
    for item in cctv.get("most_visited_brands", []):
        lines.append(f"- {item['brand']}: {item['visits']} visits")
    lines.append("")
    lines.append("### Least visited brands")
    for item in cctv.get("least_visited_brands", []):
        lines.append(f"- {item['brand']}: {item['visits']} visits")
    lines.append("")
    lines.append("### Top 5 visitor journeys")
    for journey in cctv.get("top_5_visitor_journeys", []):
        seq = " → ".join(journey["zone_sequence"])
        lines.append(f"- Visitor **{journey['visitor_id']}**: {seq}")

    lines.extend(["", "## POS Analytics", ""])
    pos = bundle.pos
    lines.append(f"- Total invoices: **{pos.get('total_invoices', 0)}**")
    lines.append(f"- Total revenue: **INR {pos.get('total_revenue', 0):,.2f}**")
    lines.append(
        f"- Average basket value: **INR {pos.get('average_basket_value', 0):,.2f}**"
    )
    lines.append("")
    lines.append("### Revenue by brand")
    for brand, revenue in pos.get("revenue_by_brand", {}).items():
        lines.append(f"- {brand}: INR {revenue:,.2f}")
    lines.append("")
    lines.append("### Revenue by category")
    for category, revenue in pos.get("revenue_by_category", {}).items():
        lines.append(f"- {category}: INR {revenue:,.2f}")
    lines.append("")
    lines.append("### Top selling products")
    for item in pos.get("top_selling_products", [])[:5]:
        lines.append(
            f"- ({item['invoice_line_count']}×) {item['product'][:100]}..."
            if len(item["product"]) > 100
            else f"- ({item['invoice_line_count']}×) {item['product']}"
        )

    lines.extend(["", "## Purchase Matching", ""])
    matching = bundle.matching
    lines.append(f"- Matched invoices: **{matching.get('matched_invoices', 0)}**")
    lines.append(f"- Unmatched invoices: **{matching.get('unmatched_invoices', 0)}**")
    lines.append(
        f"- Average confidence: **{matching.get('average_confidence') or 'N/A'}**"
    )

    lines.extend(["", "## Combined Insights", ""])
    for observation in bundle.combined.get("observations", []):
        lines.append(f"- {observation}")

    lines.extend(["", "## Charts", ""])
    for chart in bundle.chart_paths:
        lines.append(f"![{Path(chart).stem}]({chart})")

    return "\n".join(lines) + "\n"


def run_analytics() -> AnalyticsBundle:
    bundle = AnalyticsBundle()

    events, source_counts = load_all_cctv_events()
    bundle.cctv = compute_cctv_analytics(events)
    bundle.cctv["event_sources"] = source_counts

    transactions = load_json_array(TRANSACTIONS_PATH)
    bundle.pos = compute_pos_analytics(transactions)

    matches = load_json_array(PURCHASE_MATCHES_PATH)
    bundle.matching = compute_matching_analytics(matches)

    bundle.combined = compute_combined_insights(
        bundle.cctv, bundle.pos, bundle.matching
    )
    bundle.executive_insights = generate_executive_insights(bundle)
    generate_charts(bundle)

    return bundle


def main() -> None:
    bundle = run_analytics()

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "cctv_analytics": bundle.cctv,
        "pos_analytics": bundle.pos,
        "purchase_matching_analytics": bundle.matching,
        "combined_insights": bundle.combined,
        "executive_insights": bundle.executive_insights,
        "charts": bundle.chart_paths,
    }

    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    REPORT_PATH.write_text(build_markdown_report(bundle), encoding="utf-8")

    print("Retail analytics complete")
    print("=" * 50)
    print(f"CCTV events: {bundle.cctv.get('total_events', 0)}")
    print(f"POS invoices: {bundle.pos.get('total_invoices', 0)}")
    print(f"Matched invoices: {bundle.matching.get('matched_invoices', 0)}")
    print(f"Charts saved: {len(bundle.chart_paths)} in {CHARTS_DIR.name}/")
    print(f"\nSummary: {SUMMARY_PATH}")
    print(f"Report:  {REPORT_PATH}")
    print("\nTop insights:")
    for idx, insight in enumerate(bundle.executive_insights, start=1):
        print(f"  {idx}. {insight}")


if __name__ == "__main__":
    main()
