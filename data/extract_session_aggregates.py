#!/usr/bin/env python3
"""
Extract per-session aggregate statistics from prompts_unified.csv.

Produces sessions_aggregate.csv with numeric statistics only (no prompt text)
for independent verification of paper claims (Section 4).

Session IDs are anonymized via SHA-256 hashing.

Usage:
    python extract_session_aggregates.py --input prompts_unified.csv --output samples/sessions_aggregate.csv
    python extract_session_aggregates.py --stats  # Print summary statistics
"""

import argparse
import csv
import hashlib
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path


def hash_session_id(session_id: str, salt: str = "codified-context-2026") -> str:
    """Anonymize session UUID to 12-char hex hash."""
    h = hashlib.sha256(f"{salt}:{session_id}".encode()).hexdigest()
    return h[:12]


def safe_int(value: str, default: int = 0) -> int:
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def safe_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def aggregate_sessions(input_path: Path) -> list[dict]:
    """Read prompts CSV and aggregate by session_id."""
    sessions = defaultdict(list)

    with open(input_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row.get("session_id", "").strip()
            if not sid:
                continue
            sessions[sid].append(row)

    results = []
    for sid, rows in sessions.items():
        word_counts = [safe_int(r["word_count"]) for r in rows]
        categories = Counter(r["category"] for r in rows if r.get("category"))

        # Collect unique agent types across all rows
        agent_types_set = set()
        for r in rows:
            types = r.get("agent_types", "").strip()
            if types:
                for t in types.split(","):
                    t = t.strip()
                    if t:
                        agent_types_set.add(t)

        # Determine primary model (most common non-empty)
        models = [r.get("model", "") for r in rows if r.get("model", "").strip()]
        primary_model = Counter(models).most_common(1)[0][0] if models else ""

        # Earliest date in session
        dates = sorted(r["date"] for r in rows if r.get("date"))
        date = dates[0] if dates else ""

        # Platform (most common)
        platforms = [r.get("platform", "") for r in rows if r.get("platform", "").strip()]
        platform = Counter(platforms).most_common(1)[0][0] if platforms else ""

        results.append({
            "session_id": hash_session_id(sid),
            "date": date,
            "prompt_count": len(rows),
            "total_word_count": sum(word_counts),
            "median_word_count": int(statistics.median(word_counts)) if word_counts else 0,
            "category_distribution": json.dumps(dict(categories.most_common()), separators=(",", ":")),
            "total_tokens_in": sum(safe_int(r.get("response_tokens_in")) for r in rows),
            "total_tokens_out": sum(safe_int(r.get("response_tokens_out")) for r in rows),
            "total_cache_read": sum(safe_int(r.get("response_cache_read")) for r in rows),
            "total_cache_create": sum(safe_int(r.get("response_cache_create")) for r in rows),
            "total_cost_usd": round(sum(safe_float(r.get("cost_total_usd")) for r in rows), 2),
            "total_tool_calls": sum(safe_int(r.get("tool_count")) for r in rows),
            "total_agents_spawned": sum(safe_int(r.get("agents_spawned")) for r in rows),
            "agent_types_used": ",".join(sorted(agent_types_set)),
            "model": primary_model,
            "platform": platform,
        })

    # Sort by date
    results.sort(key=lambda r: r["date"])
    return results


def write_csv(results: list[dict], output_path: Path):
    """Write aggregated results to CSV."""
    if not results:
        print("No results to write.", file=sys.stderr)
        return

    fieldnames = list(results[0].keys())
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Wrote {len(results)} sessions to {output_path}")


def print_stats(results: list[dict]):
    """Print summary statistics for verification against paper claims."""
    print(f"\n{'='*60}")
    print("SESSION AGGREGATE STATISTICS")
    print(f"{'='*60}")
    print(f"Total sessions: {len(results)}")

    total_prompts = sum(r["prompt_count"] for r in results)
    print(f"Total prompts: {total_prompts}")

    all_medians = [r["median_word_count"] for r in results]
    print(f"Overall median prompt length: {statistics.median(all_medians):.0f} words")

    total_cost = sum(r["total_cost_usd"] for r in results)
    print(f"Total cost: ${total_cost:.2f}")
    print(f"Avg cost/session: ${total_cost / len(results):.2f}")

    total_agents = sum(r["total_agents_spawned"] for r in results)
    print(f"Total agent invocations: {total_agents}")

    total_tools = sum(r["total_tool_calls"] for r in results)
    print(f"Total tool calls: {total_tools}")

    # Category totals
    all_cats = Counter()
    for r in results:
        cats = json.loads(r["category_distribution"])
        all_cats.update(cats)
    print(f"\nTop 10 categories:")
    for cat, count in all_cats.most_common(10):
        print(f"  {cat}: {count}")

    # Date range
    dates = sorted(r["date"] for r in results if r["date"])
    if dates:
        print(f"\nDate range: {dates[0]} to {dates[-1]}")

    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Extract per-session aggregate statistics")
    parser.add_argument("--input", type=Path,
                        default=Path("prompts_unified.csv"),
                        help="Path to prompts_unified.csv")
    parser.add_argument("--output", type=Path,
                        default=Path("samples/sessions_aggregate.csv"),
                        help="Output CSV path")
    parser.add_argument("--stats", action="store_true",
                        help="Print summary statistics")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: {args.input} not found", file=sys.stderr)
        sys.exit(1)

    results = aggregate_sessions(args.input)
    write_csv(results, args.output)

    if args.stats:
        print_stats(results)


if __name__ == "__main__":
    main()
