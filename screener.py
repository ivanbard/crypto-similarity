from __future__ import annotations

import ast
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

COINGECKO_API_KEY = os.getenv("COINGECKO_KEY") or os.getenv("COINGECKO-KEY")
RAW_OUTPUT_PATH = Path("data/raw/crypto_features_raw.csv")
PROCESSED_OUTPUT_PATH = Path("data/processed/screener.json")
HTML_OUTPUT_PATH = Path("dashboard/index.html")
SNAPSHOT_OUTPUT_DIR = Path("data/history/snapshots")
VALIDATION_OUTPUT_DIR = Path("data/history/reports")

BUNDLE_RULES = [
    {
        "id": "layer1",
        "label": "Layer 1",
        "description": "Base-layer smart contract networks and settlement chains.",
        "keywords": ["layer 1 (l1)", "smart contract platform", "proof of work (pow)", "proof of stake (pos)"],
    },
    {
        "id": "layer2",
        "label": "Layer 2",
        "description": "Scaling layers and rollup-style execution networks.",
        "keywords": ["layer 2 (l2)", "rollup", "zkevm", "optimistic"],
    },
    {
        "id": "stablecoins",
        "label": "Stablecoins",
        "description": "Fiat-backed and crypto-backed stability assets.",
        "keywords": ["stablecoin", "stablecoins"],
    },
    {
        "id": "defi",
        "label": "DeFi",
        "description": "Protocols for decentralized trading, lending, and on-chain yield.",
        "keywords": [
            "decentralized finance (defi)",
            "lending",
            "borrowing",
            "yield",
            "decentralized exchange (dex)",
            "automated market maker",
        ],
    },
    {
        "id": "exchange",
        "label": "Exchange Tokens",
        "description": "Tokens tied to centralized exchange platforms.",
        "keywords": ["exchange-based tokens", "centralized exchange (cex) token"],
    },
    {
        "id": "liquid-staking",
        "label": "Liquid Staking",
        "description": "Staking derivatives and liquid restaking assets.",
        "keywords": ["liquid staking", "liquid staked", "restaking"],
    },
    {
        "id": "rwa",
        "label": "Real-World Assets",
        "description": "Tokens representing or servicing off-chain real-world assets.",
        "keywords": ["real world assets (rwa)", "rwa"],
    },
    {
        "id": "oracle",
        "label": "Oracles",
        "description": "Projects that supply external data into crypto networks.",
        "keywords": ["oracle", "oracles"],
    },
    {
        "id": "ai",
        "label": "AI",
        "description": "AI-linked tokens, agent frameworks, and compute networks.",
        "keywords": ["artificial intelligence", "ai", "agent", "agents", "depin"],
    },
    {
        "id": "gaming",
        "label": "Gaming",
        "description": "Gaming, gamefi, and metaverse-oriented assets.",
        "keywords": ["gaming", "gamefi", "metaverse"],
    },
    {
        "id": "meme",
        "label": "Meme",
        "description": "Meme and culture-driven speculative assets.",
        "keywords": ["meme", "dog-themed", "cat-themed", "frog-themed", "elon musk-inspired", "4chan-themed"],
    },
    {
        "id": "privacy",
        "label": "Privacy",
        "description": "Assets focused on private transfers and privacy-preserving systems.",
        "keywords": ["privacy", "privacy coins"],
    },
    {
        "id": "payments",
        "label": "Payments",
        "description": "Networks and assets positioned around transfer rails or remittance.",
        "keywords": ["payments", "payment", "remittance", "xrp ledger"],
    },
    {
        "id": "infrastructure",
        "label": "Infrastructure",
        "description": "Data, storage, interoperability, and general crypto middleware.",
        "keywords": ["interoperability", "storage", "data availability", "modular", "infrastructure"],
    },
]
BUNDLE_LOOKUP = {rule["id"]: rule for rule in BUNDLE_RULES}


@dataclass(frozen=True)
class FilterConfig:
    min_market_cap_usd: float = 10_000_000
    min_volume_usd: float = 500_000
    min_age_days: int = 180
    min_peer_count: int = 3
    max_peer_count: int = 3
    min_gap_ratio: float = 1.5


MARKET_COLUMNS = [
    "id",
    "symbol",
    "name",
    "current_price",
    "market_cap",
    "total_volume",
    "price_change_percentage_24h",
]

DETAIL_COLUMNS = [
    "id",
    "categories",
    "hashing_algorithm",
    "genesis_date",
    "sentiment_up",
    "sentiment_down",
    "developer_score",
    "community_score",
    "liquidity_score",
]


def get_coingecko_headers() -> dict[str, str]:
    headers: dict[str, str] = {}
    if COINGECKO_API_KEY:
        headers["x-cg-demo-api-key"] = COINGECKO_API_KEY
    return headers


def fetch_coingecko_markets(pages: int = 3, per_page: int = 250, sleep_seconds: float = 1.2) -> pd.DataFrame:
    all_rows: list[dict[str, Any]] = []

    for page in range(1, pages + 1):
        response = requests.get(
            "https://api.coingecko.com/api/v3/coins/markets",
            params={
                "vs_currency": "usd",
                "per_page": per_page,
                "page": page,
                "order": "market_cap_desc",
                "price_change_percentage": "24h",
            },
            headers=get_coingecko_headers(),
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        all_rows.extend(payload)
        time.sleep(sleep_seconds)

    return pd.DataFrame(all_rows)


def fetch_coingecko_details(coin_id: str) -> dict[str, Any]:
    response = requests.get(
        f"https://api.coingecko.com/api/v3/coins/{coin_id}",
        params={
            "localization": "false",
            "tickers": "false",
            "market_data": "false",
            "community_data": "true",
            "developer_data": "true",
            "sparkline": "false",
        },
        headers=get_coingecko_headers(),
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    return {
        "id": coin_id,
        "categories": payload.get("categories", []),
        "hashing_algorithm": payload.get("hashing_algorithm"),
        "genesis_date": payload.get("genesis_date"),
        "sentiment_up": payload.get("sentiment_votes_up_percentage"),
        "sentiment_down": payload.get("sentiment_votes_down_percentage"),
        "developer_score": payload.get("developer_score"),
        "community_score": payload.get("community_score"),
        "liquidity_score": payload.get("liquidity_score"),
    }


def collect_market_dataset(pages: int = 3, sleep_seconds: float = 1.2) -> pd.DataFrame:
    market_df = fetch_coingecko_markets(pages=pages, sleep_seconds=sleep_seconds)
    coin_ids = market_df["id"].tolist()
    detail_rows: list[dict[str, Any]] = []

    for coin_id in coin_ids:
        try:
            detail_rows.append(fetch_coingecko_details(coin_id))
        except requests.RequestException as exc:
            print(f"warning: failed to fetch details for {coin_id}: {exc}")
        time.sleep(sleep_seconds)

    detail_df = pd.DataFrame(detail_rows)
    final_df = market_df[MARKET_COLUMNS].copy()
    if not detail_df.empty:
        final_df = final_df.merge(detail_df[DETAIL_COLUMNS], on="id", how="left")
    return final_df.sort_values("market_cap", ascending=False).reset_index(drop=True)


def save_raw_dataset(df: pd.DataFrame, output_path: Path = RAW_OUTPUT_PATH) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def parse_categories(raw_value: Any) -> list[str]:
    if isinstance(raw_value, list):
        return [str(item).strip() for item in raw_value if str(item).strip()]
    if raw_value is None or (isinstance(raw_value, float) and pd.isna(raw_value)):
        return []
    if not isinstance(raw_value, str):
        return []

    text = raw_value.strip()
    if not text:
        return []

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except (ValueError, SyntaxError):
        pass

    return [segment.strip() for segment in text.split("|") if segment.strip()]


def parse_genesis_date(value: Any) -> date | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, date):
        return value
    if not isinstance(value, str) or not value.strip():
        return None

    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            parsed = datetime.strptime(value.strip(), fmt)
            return parsed.date()
        except ValueError:
            continue
    return None


def compute_age_days(genesis_date: Any, today: date | None = None) -> int | None:
    parsed = parse_genesis_date(genesis_date)
    if not parsed:
        return None
    current_day = today or date.today()
    return max((current_day - parsed).days, 0)


def normalize_match_text(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
    return f" {normalized} " if normalized else " "


def matches_keywords(text: str, keywords: list[str]) -> bool:
    normalized_text = normalize_match_text(text)
    return any(normalize_match_text(keyword) in normalized_text for keyword in keywords)


def assign_bundles(categories: list[str], hashing_algorithm: Any = None) -> list[str]:
    haystack = [*categories]
    if isinstance(hashing_algorithm, str) and hashing_algorithm.strip():
        haystack.append(hashing_algorithm.strip())
    text = " | ".join(haystack)

    matches: list[str] = []
    for rule in BUNDLE_RULES:
        if matches_keywords(text, rule["keywords"]):
            matches.append(rule["id"])
    return matches


def matched_features(categories: list[str], bundle_id: str) -> list[str]:
    rule = BUNDLE_LOOKUP[bundle_id]
    features = [
        category
        for category in categories
        if matches_keywords(category, rule["keywords"])
    ]
    return features[:4] or [rule["label"]]


def prepare_asset_frame(raw_df: pd.DataFrame, config: FilterConfig) -> pd.DataFrame:
    df = raw_df.copy()
    for column in ["current_price", "market_cap", "total_volume", "price_change_percentage_24h"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["parsed_categories"] = df.get("categories", pd.Series(dtype=object)).apply(parse_categories)
    df["age_days"] = df.get("genesis_date", pd.Series(dtype=object)).apply(compute_age_days)
    df["bundle_ids"] = df.apply(
        lambda row: assign_bundles(row["parsed_categories"], row.get("hashing_algorithm")),
        axis=1,
    )

    def build_flags(row: pd.Series) -> list[str]:
        flags: list[str] = []
        if pd.isna(row.get("market_cap")):
            flags.append("missing-market-cap")
        elif float(row["market_cap"]) < config.min_market_cap_usd:
            flags.append("small-market-cap")

        if pd.isna(row.get("total_volume")) or float(row["total_volume"]) < config.min_volume_usd:
            flags.append("low-volume")

        age_days = row.get("age_days")
        if age_days is None or (isinstance(age_days, float) and pd.isna(age_days)):
            flags.append("unknown-age")
        elif int(age_days) < config.min_age_days:
            flags.append("young-asset")

        if not row["bundle_ids"]:
            flags.append("uncategorized")

        return flags

    df["risk_flags"] = df.apply(build_flags, axis=1)
    blocking_flags = {"missing-market-cap", "small-market-cap", "low-volume", "young-asset"}
    df["passes_basic_filters"] = df["risk_flags"].apply(lambda flags: not any(flag in blocking_flags for flag in flags))
    return df


def expand_bundle_memberships(asset_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for asset in asset_df.itertuples(index=False):
        for bundle_id in asset.bundle_ids:
            rule = BUNDLE_LOOKUP[bundle_id]
            rows.append(
                {
                    "bundle_id": bundle_id,
                    "bundle_label": rule["label"],
                    "bundle_description": rule["description"],
                    "asset_id": asset.id,
                    "symbol": asset.symbol,
                    "name": asset.name,
                    "current_price": float(asset.current_price) if pd.notna(asset.current_price) else None,
                    "market_cap": float(asset.market_cap) if pd.notna(asset.market_cap) else None,
                    "total_volume": float(asset.total_volume) if pd.notna(asset.total_volume) else None,
                    "price_change_percentage_24h": float(asset.price_change_percentage_24h) if pd.notna(asset.price_change_percentage_24h) else None,
                    "age_days": int(asset.age_days) if asset.age_days is not None and pd.notna(asset.age_days) else None,
                    "risk_flags": list(asset.risk_flags),
                    "passes_basic_filters": bool(asset.passes_basic_filters),
                    "matched_features": matched_features(asset.parsed_categories, bundle_id),
                    "raw_categories": list(asset.parsed_categories[:6]),
                }
            )

    return pd.DataFrame(rows)


def _format_gap(gap_ratio: float) -> str:
    return f"{gap_ratio:.1f}x"


def build_bundle_payload(asset_df: pd.DataFrame, config: FilterConfig) -> list[dict[str, Any]]:
    bundle_df = expand_bundle_memberships(asset_df)
    if bundle_df.empty:
        return []

    bundles: list[dict[str, Any]] = []
    minimum_assets = max(config.min_peer_count + 1, 4)

    for bundle_id, group in bundle_df.groupby("bundle_id", sort=False):
        group = group.sort_values("market_cap", ascending=False, na_position="last").reset_index(drop=True)
        qualified = group[group["passes_basic_filters"]].copy()
        candidates: list[dict[str, Any]] = []

        if len(qualified) >= minimum_assets:
            for asset in qualified.sort_values("market_cap", ascending=True).itertuples(index=False):
                larger_peers = qualified[qualified["market_cap"] > asset.market_cap].sort_values("market_cap", ascending=True).head(config.max_peer_count)
                if len(larger_peers) < config.min_peer_count:
                    continue

                reference_cap = float(larger_peers["market_cap"].median())
                gap_ratio = reference_cap / float(asset.market_cap)
                if gap_ratio < config.min_gap_ratio:
                    continue

                peer_assets = [
                    {
                        "id": peer.asset_id,
                        "name": peer.name,
                        "symbol": peer.symbol.upper(),
                        "market_cap": float(peer.market_cap),
                    }
                    for peer in larger_peers.itertuples(index=False)
                ]
                candidates.append(
                    {
                        "id": asset.asset_id,
                        "name": asset.name,
                        "symbol": asset.symbol.upper(),
                        "current_price": asset.current_price,
                        "market_cap": float(asset.market_cap),
                        "total_volume": float(asset.total_volume),
                        "age_days": asset.age_days,
                        "price_change_percentage_24h": asset.price_change_percentage_24h,
                        "score": round(gap_ratio, 3),
                        "gap_ratio": round(gap_ratio, 3),
                        "peer_market_cap": reference_cap,
                        "peer_assets": peer_assets,
                        "matched_features": list(asset.matched_features),
                        "raw_categories": list(asset.raw_categories),
                        "risk_flags": [flag for flag in asset.risk_flags if flag != "uncategorized"],
                        "explanation": (
                            f"{asset.name} sits around {_format_gap(gap_ratio)} below the median market cap of its next larger "
                            f"{BUNDLE_LOOKUP[bundle_id]['label']} peers."
                        ),
                    }
                )

        candidates.sort(key=lambda item: item["score"], reverse=True)
        bundles.append(
            {
                "id": bundle_id,
                "label": BUNDLE_LOOKUP[bundle_id]["label"],
                "description": BUNDLE_LOOKUP[bundle_id]["description"],
                "asset_count": int(len(group)),
                "eligible_asset_count": int(len(qualified)),
                "candidate_count": int(len(candidates)),
                "median_market_cap": float(qualified["market_cap"].median()) if not qualified.empty else None,
                "top_gap_ratio": candidates[0]["gap_ratio"] if candidates else None,
                "candidates": candidates,
            }
        )

    bundles.sort(
        key=lambda item: (
            item["candidate_count"],
            item["top_gap_ratio"] or 0,
            item["eligible_asset_count"],
        ),
        reverse=True,
    )
    return bundles


def build_screener_payload(raw_df: pd.DataFrame, config: FilterConfig | None = None) -> dict[str, Any]:
    active_config = config or FilterConfig()
    prepared_df = prepare_asset_frame(raw_df, active_config)
    bundles = build_bundle_payload(prepared_df, active_config)
    eligible_assets = prepared_df[
        prepared_df["passes_basic_filters"] & prepared_df["bundle_ids"].apply(bool)
    ]

    return {
        "generated_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "config": asdict(active_config),
        "summary": {
            "total_assets": int(len(prepared_df)),
            "eligible_assets": int(len(eligible_assets)),
            "bundle_count": int(len(bundles)),
            "candidate_count": int(sum(bundle["candidate_count"] for bundle in bundles)),
        },
        "bundles": bundles,
    }


def build_snapshot_frame(
    raw_df: pd.DataFrame,
    config: FilterConfig | None = None,
    snapshot_date: date | None = None,
) -> pd.DataFrame:
    active_config = config or FilterConfig()
    snapshot_day = snapshot_date or date.today()
    prepared_df = prepare_asset_frame(raw_df, active_config)
    bundle_payload = build_bundle_payload(prepared_df, active_config)
    bundle_df = expand_bundle_memberships(prepared_df)
    eligible_df = bundle_df[bundle_df["passes_basic_filters"]].copy()

    columns = [
        "snapshot_date",
        "asset_id",
        "symbol",
        "name",
        "bundle_id",
        "bundle_label",
        "is_candidate",
        "rank",
        "gap_ratio",
        "current_price",
        "market_cap",
        "total_volume",
    ]
    if eligible_df.empty:
        return pd.DataFrame(columns=columns)

    candidate_lookup: dict[tuple[str, str], dict[str, float | int]] = {}
    for bundle in bundle_payload:
        for rank, candidate in enumerate(bundle["candidates"], start=1):
            candidate_lookup[(bundle["id"], candidate["id"])] = {
                "rank": rank,
                "gap_ratio": candidate["gap_ratio"],
            }

    snapshot_rows: list[dict[str, Any]] = []
    for row in eligible_df.itertuples(index=False):
        candidate_meta = candidate_lookup.get((row.bundle_id, row.asset_id), {})
        snapshot_rows.append(
            {
                "snapshot_date": snapshot_day.isoformat(),
                "asset_id": row.asset_id,
                "symbol": row.symbol,
                "name": row.name,
                "bundle_id": row.bundle_id,
                "bundle_label": row.bundle_label,
                "is_candidate": (row.bundle_id, row.asset_id) in candidate_lookup,
                "rank": candidate_meta.get("rank"),
                "gap_ratio": candidate_meta.get("gap_ratio"),
                "current_price": row.current_price,
                "market_cap": row.market_cap,
                "total_volume": row.total_volume,
            }
        )

    snapshot_df = pd.DataFrame(snapshot_rows, columns=columns)
    if not snapshot_df.empty:
        snapshot_df["rank"] = snapshot_df["rank"].astype("Int64")
    return snapshot_df.sort_values(["bundle_label", "rank", "market_cap"], ascending=[True, True, False], na_position="last").reset_index(drop=True)


def save_snapshot_frame(
    snapshot_df: pd.DataFrame,
    output_dir: Path = SNAPSHOT_OUTPUT_DIR,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_value = snapshot_df["snapshot_date"].iloc[0] if not snapshot_df.empty else date.today().isoformat()
    output_path = output_dir / f"{snapshot_value}.csv"
    snapshot_df.to_csv(output_path, index=False)
    return output_path


def load_snapshot_history(snapshot_dir: Path = SNAPSHOT_OUTPUT_DIR) -> pd.DataFrame:
    snapshot_files = sorted(snapshot_dir.glob("*.csv"))
    if not snapshot_files:
        return pd.DataFrame(
            columns=[
                "snapshot_date",
                "asset_id",
                "symbol",
                "name",
                "bundle_id",
                "bundle_label",
                "is_candidate",
                "rank",
                "gap_ratio",
                "current_price",
                "market_cap",
                "total_volume",
            ]
        )

    frames = [pd.read_csv(path) for path in snapshot_files]
    history_df = pd.concat(frames, ignore_index=True)
    history_df["snapshot_date"] = pd.to_datetime(history_df["snapshot_date"]).dt.date
    history_df["is_candidate"] = history_df["is_candidate"].astype(str).str.lower().isin({"true", "1"})
    for column in ["rank", "gap_ratio", "current_price", "market_cap", "total_volume"]:
        history_df[column] = pd.to_numeric(history_df[column], errors="coerce")
    return history_df


def build_validation_report(
    snapshot_history: pd.DataFrame,
    horizon_days: int = 30,
    min_benchmark_assets: int = 3,
) -> dict[str, Any]:
    if snapshot_history.empty:
        raise ValueError("No snapshot history available for validation")

    history_df = snapshot_history.copy()
    history_df["snapshot_date"] = pd.to_datetime(history_df["snapshot_date"]).dt.date
    history_df["is_candidate"] = history_df["is_candidate"].astype(bool)
    for column in ["rank", "gap_ratio", "current_price", "market_cap", "total_volume"]:
        history_df[column] = pd.to_numeric(history_df[column], errors="coerce")

    price_rows = history_df[
        ["snapshot_date", "bundle_id", "bundle_label", "asset_id", "current_price"]
    ].drop_duplicates()
    entry_prices = price_rows.rename(columns={"current_price": "entry_price"}).copy()
    entry_prices["exit_date"] = entry_prices["snapshot_date"].apply(lambda value: value + timedelta(days=horizon_days))
    exit_prices = price_rows.rename(columns={"snapshot_date": "exit_date", "current_price": "exit_price"})

    returns_df = entry_prices.merge(
        exit_prices[["exit_date", "bundle_id", "asset_id", "exit_price"]],
        on=["exit_date", "bundle_id", "asset_id"],
        how="inner",
    )
    returns_df["forward_return"] = (returns_df["exit_price"] / returns_df["entry_price"]) - 1

    benchmark_df = returns_df.groupby(["snapshot_date", "bundle_id"], as_index=False).agg(
        bundle_median_return=("forward_return", "median"),
        benchmark_asset_count=("asset_id", "nunique"),
    )

    candidate_meta = history_df[history_df["is_candidate"]].drop_duplicates(
        subset=["snapshot_date", "bundle_id", "asset_id"]
    )
    candidate_details = candidate_meta.merge(
        returns_df[
            [
                "snapshot_date",
                "bundle_id",
                "bundle_label",
                "asset_id",
                "entry_price",
                "exit_date",
                "exit_price",
                "forward_return",
            ]
        ],
        on=["snapshot_date", "bundle_id", "bundle_label", "asset_id"],
        how="inner",
    ).merge(
        benchmark_df,
        on=["snapshot_date", "bundle_id"],
        how="inner",
    )

    candidate_details = candidate_details[
        candidate_details["benchmark_asset_count"] >= min_benchmark_assets
    ].copy()
    candidate_details["excess_return"] = candidate_details["forward_return"] - candidate_details["bundle_median_return"]
    candidate_details["beat_bundle_median"] = candidate_details["excess_return"] > 0
    candidate_details = candidate_details.sort_values(["snapshot_date", "bundle_label", "rank", "asset_id"]).reset_index(drop=True)

    if candidate_details.empty:
        overall_summary = {
            "candidate_observations": 0,
            "bundle_count_evaluated": 0,
            "hit_rate": None,
            "median_forward_return": None,
            "median_bundle_return": None,
            "median_excess_return": None,
        }
        bundle_summary = pd.DataFrame(
            columns=[
                "bundle_id",
                "bundle_label",
                "candidate_observations",
                "hit_rate",
                "median_forward_return",
                "median_bundle_return",
                "median_excess_return",
            ]
        )
    else:
        overall_summary = {
            "candidate_observations": int(len(candidate_details)),
            "bundle_count_evaluated": int(candidate_details["bundle_id"].nunique()),
            "hit_rate": round(float(candidate_details["beat_bundle_median"].mean()), 4),
            "median_forward_return": round(float(candidate_details["forward_return"].median()), 4),
            "median_bundle_return": round(float(candidate_details["bundle_median_return"].median()), 4),
            "median_excess_return": round(float(candidate_details["excess_return"].median()), 4),
        }
        bundle_summary = candidate_details.groupby(["bundle_id", "bundle_label"], as_index=False).agg(
            candidate_observations=("asset_id", "count"),
            hit_rate=("beat_bundle_median", "mean"),
            median_forward_return=("forward_return", "median"),
            median_bundle_return=("bundle_median_return", "median"),
            median_excess_return=("excess_return", "median"),
        )
        for column in ["hit_rate", "median_forward_return", "median_bundle_return", "median_excess_return"]:
            bundle_summary[column] = bundle_summary[column].round(4)

    snapshot_dates = sorted(history_df["snapshot_date"].unique())
    return {
        "generated_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "horizon_days": horizon_days,
        "min_benchmark_assets": min_benchmark_assets,
        "snapshot_count": int(len(snapshot_dates)),
        "snapshot_start_date": snapshot_dates[0].isoformat() if snapshot_dates else None,
        "snapshot_end_date": snapshot_dates[-1].isoformat() if snapshot_dates else None,
        "overall_summary": overall_summary,
        "bundle_summary": bundle_summary,
        "candidate_details": candidate_details,
    }


def save_validation_report(
    report: dict[str, Any],
    output_dir: Path = VALIDATION_OUTPUT_DIR,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    horizon_days = report["horizon_days"]
    summary_path = output_dir / f"summary_{horizon_days}d.json"
    bundle_summary_path = output_dir / f"bundle_summary_{horizon_days}d.csv"
    candidate_details_path = output_dir / f"candidate_details_{horizon_days}d.csv"

    summary_payload = {
        key: value
        for key, value in report.items()
        if key not in {"bundle_summary", "candidate_details"}
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    report["bundle_summary"].to_csv(bundle_summary_path, index=False)
    report["candidate_details"].to_csv(candidate_details_path, index=False)
    return summary_path, bundle_summary_path, candidate_details_path


def render_dashboard_html(payload: dict[str, Any]) -> str:
    payload_json = json.dumps(payload, separators=(",", ":"))
    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Crypto Similarity Screener</title>
  <style>
    :root {{
      --bg: #07111f;
      --panel: rgba(8, 20, 36, 0.82);
      --panel-strong: rgba(10, 27, 48, 0.95);
      --line: rgba(148, 185, 255, 0.18);
      --text: #ecf4ff;
      --muted: #99afc9;
      --accent: #61f0d1;
      --accent-warm: #ffb86b;
      --danger: #ff7f7f;
      --shadow: 0 24px 80px rgba(0, 0, 0, 0.35);
      --radius: 24px;
    }}

    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Avenir Next", "Segoe UI", sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(97, 240, 209, 0.16), transparent 28%),
        radial-gradient(circle at top right, rgba(255, 184, 107, 0.16), transparent 26%),
        linear-gradient(180deg, #040b14 0%, #081323 45%, #07111f 100%);
      min-height: 100vh;
    }}

    .shell {{
      width: min(1440px, calc(100vw - 32px));
      margin: 24px auto 40px;
    }}

    .hero {{
      padding: 28px;
      border: 1px solid var(--line);
      border-radius: 32px;
      background: linear-gradient(135deg, rgba(9, 22, 39, 0.94), rgba(7, 15, 27, 0.88));
      box-shadow: var(--shadow);
      overflow: hidden;
      position: relative;
    }}

    .hero::after {{
      content: "";
      position: absolute;
      inset: auto -12% -50% auto;
      width: 320px;
      height: 320px;
      border-radius: 50%;
      background: radial-gradient(circle, rgba(97, 240, 209, 0.18), transparent 68%);
      pointer-events: none;
    }}

    h1, h2, h3 {{
      font-family: "Iowan Old Style", "Palatino Linotype", serif;
      letter-spacing: 0.01em;
      margin: 0;
    }}

    .eyebrow {{
      text-transform: uppercase;
      letter-spacing: 0.24em;
      color: var(--accent);
      font-size: 12px;
      margin-bottom: 10px;
    }}

    .hero p {{
      max-width: 880px;
      color: var(--muted);
      margin: 12px 0 0;
      line-height: 1.55;
    }}

    .stat-grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 14px;
      margin-top: 24px;
    }}

    .stat-card, .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      backdrop-filter: blur(10px);
    }}

    .stat-card {{ padding: 18px 20px; }}
    .stat-label {{ color: var(--muted); font-size: 13px; text-transform: uppercase; letter-spacing: 0.08em; }}
    .stat-value {{ font-size: 30px; margin-top: 8px; }}

    .controls {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 14px;
      margin: 18px 0;
    }}

    .control {{ padding: 18px; }}
    .control label {{ display: block; color: var(--muted); font-size: 12px; margin-bottom: 10px; text-transform: uppercase; letter-spacing: 0.08em; }}
    .control select, .control input {{
      width: 100%;
      border-radius: 16px;
      border: 1px solid rgba(148, 185, 255, 0.26);
      background: rgba(5, 13, 23, 0.82);
      color: var(--text);
      padding: 12px 14px;
      font: inherit;
    }}

    .layout {{
      display: grid;
      grid-template-columns: 320px minmax(0, 1fr);
      gap: 18px;
      align-items: start;
    }}

    .bundle-list {{ padding: 18px; position: sticky; top: 16px; }}
    .bundle-item {{
      display: block;
      width: 100%;
      background: transparent;
      border: 1px solid transparent;
      color: inherit;
      text-align: left;
      padding: 16px;
      border-radius: 18px;
      margin-bottom: 10px;
      cursor: pointer;
      transition: 140ms ease;
    }}

    .bundle-item:hover, .bundle-item.active {{
      border-color: rgba(97, 240, 209, 0.38);
      background: rgba(97, 240, 209, 0.08);
      transform: translateY(-1px);
    }}

    .bundle-title {{ font-size: 18px; }}
    .bundle-meta {{ margin-top: 6px; color: var(--muted); font-size: 13px; }}

    .results {{ padding: 22px; }}
    .results-header {{ display: flex; justify-content: space-between; gap: 16px; align-items: start; margin-bottom: 20px; }}
    .results-header p {{ margin: 8px 0 0; color: var(--muted); line-height: 1.5; }}

    .pill-row {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 18px; }}
    .pill {{
      border-radius: 999px;
      padding: 8px 12px;
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid rgba(148, 185, 255, 0.18);
      color: var(--muted);
      font-size: 13px;
    }}

    table {{ width: 100%; border-collapse: collapse; margin-top: 18px; }}
    th, td {{ padding: 14px 10px; border-bottom: 1px solid rgba(148, 185, 255, 0.12); text-align: left; vertical-align: top; }}
    th {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; }}
    td strong {{ display: block; font-size: 15px; }}
    td span {{ color: var(--muted); font-size: 13px; }}

    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 14px;
      margin-top: 18px;
    }}

    .candidate-card {{ padding: 18px; }}
    .candidate-card h3 {{ font-size: 24px; }}
    .candidate-card p {{ color: var(--muted); line-height: 1.5; }}
    .candidate-card .gap {{ color: var(--accent-warm); font-size: 28px; margin: 10px 0 4px; }}
    .peer-list, .feature-list {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }}
    .peer, .feature {{
      padding: 7px 10px;
      border-radius: 999px;
      font-size: 12px;
      background: rgba(97, 240, 209, 0.08);
      border: 1px solid rgba(97, 240, 209, 0.2);
    }}

    .empty {{
      margin-top: 18px;
      padding: 28px;
      border-radius: 22px;
      background: rgba(255, 255, 255, 0.03);
      border: 1px dashed rgba(148, 185, 255, 0.2);
      color: var(--muted);
    }}

    .down {{ color: var(--danger); }}
    .up {{ color: var(--accent); }}

    @media (max-width: 1080px) {{
      .layout {{ grid-template-columns: 1fr; }}
      .bundle-list {{ position: static; }}
      .controls, .stat-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    }}

    @media (max-width: 720px) {{
      .shell {{ width: min(100vw - 18px, 100%); margin: 12px auto 24px; }}
      .hero, .results, .bundle-list, .control, .stat-card {{ padding: 18px; border-radius: 22px; }}
      .controls, .stat-grid {{ grid-template-columns: 1fr; }}
      .results-header {{ flex-direction: column; }}
      table, thead, tbody, th, td, tr {{ display: block; }}
      thead {{ display: none; }}
      tr {{ padding: 12px 0; border-bottom: 1px solid rgba(148, 185, 255, 0.12); }}
      td {{ border: 0; padding: 6px 0; }}
      td::before {{ content: attr(data-label); display: block; color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px; }}
    }}
  </style>
</head>
<body>
  <div class=\"shell\">
    <section class=\"hero\">
      <div class=\"eyebrow\">Crypto Similarity Screener</div>
      <h1>Find smaller coins sitting inside stronger peer sets.</h1>
      <p>Each bundle groups assets by normalized CoinGecko features, filters out low-signal names, and ranks the survivors by how far below their next larger peers they trade on market cap.</p>
      <div class=\"stat-grid\" id=\"stats\"></div>
    </section>

    <section class=\"controls\">
      <div class=\"panel control\">
        <label for=\"sortBy\">Sort Candidates</label>
        <select id=\"sortBy\">
          <option value=\"score\">Peer Gap</option>
          <option value=\"market_cap\">Market Cap</option>
          <option value=\"total_volume\">24h Volume</option>
          <option value=\"price_change_percentage_24h\">24h Price Change</option>
        </select>
      </div>
      <div class=\"panel control\">
        <label for=\"minVolume\">Minimum 24h Volume</label>
        <input id=\"minVolume\" type=\"number\" min=\"0\" step=\"100000\" />
      </div>
      <div class=\"panel control\">
        <label for=\"minAge\">Minimum Age (days)</label>
        <input id=\"minAge\" type=\"number\" min=\"0\" step=\"30\" />
      </div>
      <div class=\"panel control\">
        <label for=\"capBand\">Market Cap Band</label>
        <select id=\"capBand\">
          <option value=\"all\">All</option>
          <option value=\"micro\">$10M to $100M</option>
          <option value=\"small\">$100M to $1B</option>
          <option value=\"mid\">$1B to $10B</option>
          <option value=\"large\">$10B+</option>
        </select>
      </div>
    </section>

    <section class=\"layout\">
      <aside class=\"panel bundle-list\">
        <h2>Bundles</h2>
        <div id=\"bundleList\"></div>
      </aside>

      <main class=\"panel results\">
        <div class=\"results-header\" id=\"bundleHeader\"></div>
        <div id=\"candidateTable\"></div>
        <div class=\"cards\" id=\"candidateCards\"></div>
      </main>
    </section>
  </div>

  <script>
    const PAYLOAD = {payload_json};
    const state = {{
      selectedBundle: PAYLOAD.bundles[0] ? PAYLOAD.bundles[0].id : null,
      sortBy: 'score',
      minVolume: PAYLOAD.config.min_volume_usd,
      minAge: PAYLOAD.config.min_age_days,
      capBand: 'all',
    }};

    const formatCompact = (value) => new Intl.NumberFormat('en-US', {{ notation: 'compact', maximumFractionDigits: 1 }}).format(value || 0);
    const formatCurrency = (value) => new Intl.NumberFormat('en-US', {{ style: 'currency', currency: 'USD', maximumFractionDigits: value >= 1 ? 0 : 4 }}).format(value || 0);
    const formatPercent = (value) => `${{value >= 0 ? '+' : ''}}${{(value || 0).toFixed(2)}}%`;
    const escapeHtml = (value) => String(value)
      .replaceAll('&', '&amp;')
      .replaceAll('<', '&lt;')
      .replaceAll('>', '&gt;')
      .replaceAll('"', '&quot;')
      .replaceAll("'", '&#039;');

    function passesBand(candidate) {{
      const cap = candidate.market_cap || 0;
      if (state.capBand === 'micro') return cap >= 1e7 && cap < 1e8;
      if (state.capBand === 'small') return cap >= 1e8 && cap < 1e9;
      if (state.capBand === 'mid') return cap >= 1e9 && cap < 1e10;
      if (state.capBand === 'large') return cap >= 1e10;
      return true;
    }}

    function getBundle(bundleId) {{
      return PAYLOAD.bundles.find((bundle) => bundle.id === bundleId) || PAYLOAD.bundles[0] || null;
    }}

    function getVisibleCandidates(bundle) {{
      const candidates = [...bundle.candidates].filter((candidate) =>
        (candidate.total_volume || 0) >= state.minVolume &&
        (candidate.age_days || 0) >= state.minAge &&
        passesBand(candidate)
      );

      candidates.sort((left, right) => {{
        if (state.sortBy === 'market_cap' || state.sortBy === 'total_volume') return (right[state.sortBy] || 0) - (left[state.sortBy] || 0);
        if (state.sortBy === 'price_change_percentage_24h') return (right[state.sortBy] || 0) - (left[state.sortBy] || 0);
        return (right.score || 0) - (left.score || 0);
      }});

      return candidates;
    }}

    function renderStats() {{
      const stats = [
        ['Assets Loaded', formatCompact(PAYLOAD.summary.total_assets)],
        ['Eligible Assets', formatCompact(PAYLOAD.summary.eligible_assets)],
        ['Tracked Bundles', formatCompact(PAYLOAD.summary.bundle_count)],
        ['Candidates Surfaced', formatCompact(PAYLOAD.summary.candidate_count)],
      ];
      document.getElementById('stats').innerHTML = stats.map(([label, value]) => `
        <div class=\"stat-card\">
          <div class=\"stat-label\">${{label}}</div>
          <div class=\"stat-value\">${{value}}</div>
        </div>
      `).join('');
    }}

    function renderBundleList() {{
      const selected = state.selectedBundle;
      document.getElementById('bundleList').innerHTML = PAYLOAD.bundles.map((bundle) => `
        <button class=\"bundle-item ${{bundle.id === selected ? 'active' : ''}}\" data-bundle=\"${{bundle.id}}\">
          <div class=\"bundle-title\">${{escapeHtml(bundle.label)}}</div>
          <div class=\"bundle-meta\">${{bundle.candidate_count}} ranked candidates • ${{bundle.eligible_asset_count}} eligible assets</div>
          <div class=\"bundle-meta\">Top peer gap: ${{bundle.top_gap_ratio ? `${{bundle.top_gap_ratio.toFixed(1)}}x` : 'n/a'}}</div>
        </button>
      `).join('');

      document.querySelectorAll('[data-bundle]').forEach((button) => {{
        button.addEventListener('click', () => {{
          state.selectedBundle = button.dataset.bundle;
          render();
        }});
      }});
    }}

    function renderBundleHeader(bundle, candidates) {{
      const medianCap = bundle.median_market_cap ? formatCurrency(bundle.median_market_cap) : 'n/a';
      document.getElementById('bundleHeader').innerHTML = `
        <div>
          <h2>${{escapeHtml(bundle.label)}}</h2>
          <p>${{escapeHtml(bundle.description)}}</p>
          <div class=\"pill-row\">
            <span class=\"pill\">${{bundle.asset_count}} tagged assets</span>
            <span class=\"pill\">${{bundle.eligible_asset_count}} pass baseline filters</span>
            <span class=\"pill\">Median cap of eligible set: ${{medianCap}}</span>
            <span class=\"pill\">${{candidates.length}} candidates after your filters</span>
          </div>
        </div>
        <div class=\"pill\">Generated ${{escapeHtml(PAYLOAD.generated_at)}}</div>
      `;
    }}

    function renderTable(candidates) {{
      if (!candidates.length) {{
        document.getElementById('candidateTable').innerHTML = `<div class=\"empty\">No candidates remain after the current filters. Lower the age or volume threshold, or switch to a different bundle.</div>`;
        return;
      }}

      const rows = candidates.map((candidate, index) => `
        <tr>
          <td data-label=\"Rank\">${{index + 1}}</td>
          <td data-label=\"Asset\"><strong>${{escapeHtml(candidate.name)}}</strong><span>${{escapeHtml(candidate.symbol)}}</span></td>
          <td data-label=\"Peer Gap\">${{candidate.gap_ratio.toFixed(1)}}x</td>
          <td data-label=\"Market Cap\">${{formatCurrency(candidate.market_cap)}}</td>
          <td data-label=\"24h Volume\">${{formatCurrency(candidate.total_volume)}}</td>
          <td data-label=\"24h Change\"><span class=\"${{(candidate.price_change_percentage_24h || 0) >= 0 ? 'up' : 'down'}}\">${{formatPercent(candidate.price_change_percentage_24h || 0)}}</span></td>
          <td data-label=\"Peer Set\">${{candidate.peer_assets.map((peer) => escapeHtml(peer.name)).join(', ')}}</td>
        </tr>
      `).join('');

      document.getElementById('candidateTable').innerHTML = `
        <table>
          <thead>
            <tr>
              <th>Rank</th>
              <th>Asset</th>
              <th>Peer Gap</th>
              <th>Market Cap</th>
              <th>24h Volume</th>
              <th>24h Change</th>
              <th>Peer Set</th>
            </tr>
          </thead>
          <tbody>${{rows}}</tbody>
        </table>
      `;
    }}

    function renderCards(candidates) {{
      document.getElementById('candidateCards').innerHTML = candidates.slice(0, 6).map((candidate) => `
        <article class=\"panel candidate-card\">
          <h3>${{escapeHtml(candidate.name)}}</h3>
          <div class=\"gap\">${{candidate.gap_ratio.toFixed(1)}}x peer gap</div>
          <p>${{escapeHtml(candidate.explanation)}}</p>
          <div class=\"feature-list\">${{candidate.matched_features.map((feature) => `<span class=\"feature\">${{escapeHtml(feature)}}</span>`).join('')}}</div>
          <div class=\"peer-list\">${{candidate.peer_assets.map((peer) => `<span class=\"peer\">${{escapeHtml(peer.name)}} • ${{formatCompact(peer.market_cap)}}</span>`).join('')}}</div>
        </article>
      `).join('');
    }}

    function render() {{
      renderStats();
      renderBundleList();
      const bundle = getBundle(state.selectedBundle);
      if (!bundle) return;
      const candidates = getVisibleCandidates(bundle);
      renderBundleHeader(bundle, candidates);
      renderTable(candidates);
      renderCards(candidates);
    }}

    document.getElementById('sortBy').value = state.sortBy;
    document.getElementById('minVolume').value = state.minVolume;
    document.getElementById('minAge').value = state.minAge;
    document.getElementById('capBand').value = state.capBand;

    document.getElementById('sortBy').addEventListener('change', (event) => {{ state.sortBy = event.target.value; render(); }});
    document.getElementById('minVolume').addEventListener('input', (event) => {{ state.minVolume = Number(event.target.value || 0); render(); }});
    document.getElementById('minAge').addEventListener('input', (event) => {{ state.minAge = Number(event.target.value || 0); render(); }});
    document.getElementById('capBand').addEventListener('change', (event) => {{ state.capBand = event.target.value; render(); }});

    render();
  </script>
</body>
</html>
"""


def save_processed_outputs(
    payload: dict[str, Any],
    json_output_path: Path = PROCESSED_OUTPUT_PATH,
    html_output_path: Path = HTML_OUTPUT_PATH,
) -> tuple[Path, Path]:
    json_output_path.parent.mkdir(parents=True, exist_ok=True)
    html_output_path.parent.mkdir(parents=True, exist_ok=True)
    json_output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    html_output_path.write_text(render_dashboard_html(payload), encoding="utf-8")
    return json_output_path, html_output_path
