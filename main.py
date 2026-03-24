from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd

from screener import (
    HTML_OUTPUT_PATH,
    PROCESSED_OUTPUT_PATH,
    RAW_OUTPUT_PATH,
    SNAPSHOT_OUTPUT_DIR,
    VALIDATION_OUTPUT_DIR,
    FilterConfig,
    build_screener_payload,
    build_snapshot_frame,
    build_validation_report,
    collect_market_dataset,
    load_snapshot_history,
    save_processed_outputs,
    save_raw_dataset,
    save_snapshot_frame,
    save_validation_report,
)


def parse_iso_date(value: str) -> date:
    return date.fromisoformat(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Crypto similarity screener")
    subparsers = parser.add_subparsers(dest="command")

    fetch_parser = subparsers.add_parser("fetch", help="Pull fresh raw market and metadata from CoinGecko")
    fetch_parser.add_argument("--pages", type=int, default=6, help="How many 250-asset market pages to fetch")
    fetch_parser.add_argument("--sleep-seconds", type=float, default=1.2, help="Delay between CoinGecko requests")
    fetch_parser.add_argument("--output", type=Path, default=RAW_OUTPUT_PATH, help="Where to save the raw CSV")

    build_parser = subparsers.add_parser("build", help="Build screener outputs from a raw CSV")
    build_parser.add_argument("--input", type=Path, default=RAW_OUTPUT_PATH, help="Raw CSV to transform")
    build_parser.add_argument("--json-output", type=Path, default=PROCESSED_OUTPUT_PATH, help="Path for processed JSON")
    build_parser.add_argument("--html-output", type=Path, default=HTML_OUTPUT_PATH, help="Path for generated dashboard HTML")
    build_parser.add_argument("--save-snapshot", action="store_true", help="Persist a dated eligible/candidate snapshot alongside the build")
    build_parser.add_argument("--snapshot-date", type=parse_iso_date, help="Override snapshot date using YYYY-MM-DD")
    build_parser.add_argument("--snapshot-dir", type=Path, default=SNAPSHOT_OUTPUT_DIR, help="Directory for dated snapshot CSVs")
    build_parser.add_argument("--min-market-cap", type=float, default=FilterConfig.min_market_cap_usd, help="Minimum market cap for ranking")
    build_parser.add_argument("--min-volume", type=float, default=FilterConfig.min_volume_usd, help="Minimum 24h volume for ranking")
    build_parser.add_argument("--min-age-days", type=int, default=FilterConfig.min_age_days, help="Minimum asset age in days")
    build_parser.add_argument("--min-peer-count", type=int, default=FilterConfig.min_peer_count, help="Minimum larger peers required per candidate")
    build_parser.add_argument("--max-peer-count", type=int, default=FilterConfig.max_peer_count, help="How many larger peers to compare against")
    build_parser.add_argument("--min-gap-ratio", type=float, default=FilterConfig.min_gap_ratio, help="Minimum peer gap ratio to surface")

    refresh_parser = subparsers.add_parser("refresh", help="Fetch fresh data and rebuild the screener outputs")
    refresh_parser.add_argument("--pages", type=int, default=6, help="How many 250-asset market pages to fetch")
    refresh_parser.add_argument("--sleep-seconds", type=float, default=1.2, help="Delay between CoinGecko requests")
    refresh_parser.add_argument("--raw-output", type=Path, default=RAW_OUTPUT_PATH, help="Where to save the raw CSV")
    refresh_parser.add_argument("--json-output", type=Path, default=PROCESSED_OUTPUT_PATH, help="Path for processed JSON")
    refresh_parser.add_argument("--html-output", type=Path, default=HTML_OUTPUT_PATH, help="Path for generated dashboard HTML")
    refresh_parser.add_argument("--save-snapshot", action="store_true", help="Persist a dated eligible/candidate snapshot after refresh")
    refresh_parser.add_argument("--snapshot-date", type=parse_iso_date, help="Override snapshot date using YYYY-MM-DD")
    refresh_parser.add_argument("--snapshot-dir", type=Path, default=SNAPSHOT_OUTPUT_DIR, help="Directory for dated snapshot CSVs")
    refresh_parser.add_argument("--min-market-cap", type=float, default=FilterConfig.min_market_cap_usd)
    refresh_parser.add_argument("--min-volume", type=float, default=FilterConfig.min_volume_usd)
    refresh_parser.add_argument("--min-age-days", type=int, default=FilterConfig.min_age_days)
    refresh_parser.add_argument("--min-peer-count", type=int, default=FilterConfig.min_peer_count)
    refresh_parser.add_argument("--max-peer-count", type=int, default=FilterConfig.max_peer_count)
    refresh_parser.add_argument("--min-gap-ratio", type=float, default=FilterConfig.min_gap_ratio)

    validate_parser = subparsers.add_parser("validate", help="Evaluate stored candidate snapshots against 30-day forward returns")
    validate_parser.add_argument("--snapshot-dir", type=Path, default=SNAPSHOT_OUTPUT_DIR, help="Directory containing dated snapshot CSVs")
    validate_parser.add_argument("--output-dir", type=Path, default=VALIDATION_OUTPUT_DIR, help="Directory for validation reports")
    validate_parser.add_argument("--horizon-days", type=int, default=30, help="Forward return horizon in days")
    validate_parser.add_argument("--min-benchmark-assets", type=int, default=3, help="Minimum assets with valid exits required for a bundle benchmark")

    return parser


def config_from_args(args: argparse.Namespace) -> FilterConfig:
    return FilterConfig(
        min_market_cap_usd=args.min_market_cap,
        min_volume_usd=args.min_volume,
        min_age_days=args.min_age_days,
        min_peer_count=args.min_peer_count,
        max_peer_count=args.max_peer_count,
        min_gap_ratio=args.min_gap_ratio,
    )


def build_outputs(
    raw_input: Path,
    json_output: Path,
    html_output: Path,
    config: FilterConfig,
    save_snapshot: bool = False,
    snapshot_date: date | None = None,
    snapshot_dir: Path = SNAPSHOT_OUTPUT_DIR,
) -> None:
    if not raw_input.exists():
        raise FileNotFoundError(f"Input dataset not found: {raw_input}")

    raw_df = pd.read_csv(raw_input)
    payload = build_screener_payload(raw_df, config)
    json_path, html_path = save_processed_outputs(payload, json_output, html_output)

    print(f"saved processed payload to {json_path}")
    print(f"saved dashboard to {html_path}")
    print(f"bundles: {payload['summary']['bundle_count']}")
    print(f"ranked candidates: {payload['summary']['candidate_count']}")

    if save_snapshot:
        snapshot_df = build_snapshot_frame(raw_df, config, snapshot_date=snapshot_date)
        snapshot_path = save_snapshot_frame(snapshot_df, output_dir=snapshot_dir)
        print(f"saved snapshot to {snapshot_path}")
        print(f"snapshot rows: {len(snapshot_df)}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    command = args.command or "build"

    if command == "fetch":
        raw_df = collect_market_dataset(pages=args.pages, sleep_seconds=args.sleep_seconds)
        output_path = save_raw_dataset(raw_df, args.output)
        print(f"saved raw market dataset to {output_path}")
        print(f"rows: {len(raw_df)}")
        return

    if command == "build":
        build_outputs(
            args.input,
            args.json_output,
            args.html_output,
            config_from_args(args),
            save_snapshot=args.save_snapshot,
            snapshot_date=args.snapshot_date,
            snapshot_dir=args.snapshot_dir,
        )
        return

    if command == "refresh":
        raw_df = collect_market_dataset(pages=args.pages, sleep_seconds=args.sleep_seconds)
        output_path = save_raw_dataset(raw_df, args.raw_output)
        print(f"saved raw market dataset to {output_path}")
        build_outputs(
            output_path,
            args.json_output,
            args.html_output,
            config_from_args(args),
            save_snapshot=args.save_snapshot,
            snapshot_date=args.snapshot_date,
            snapshot_dir=args.snapshot_dir,
        )
        return

    if command == "validate":
        snapshot_history = load_snapshot_history(args.snapshot_dir)
        report = build_validation_report(
            snapshot_history,
            horizon_days=args.horizon_days,
            min_benchmark_assets=args.min_benchmark_assets,
        )
        summary_path, bundle_summary_path, candidate_details_path = save_validation_report(report, args.output_dir)
        print(f"saved validation summary to {summary_path}")
        print(f"saved bundle validation table to {bundle_summary_path}")
        print(f"saved candidate validation table to {candidate_details_path}")
        print(f"candidate observations: {report['overall_summary']['candidate_observations']}")
        print(f"hit rate: {report['overall_summary']['hit_rate']}")
        return

    parser.error(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
