from __future__ import annotations

import ast
import json
import math
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


def fetch_coingecko_markets(pages: int = 6, per_page: int = 250, sleep_seconds: float = 1.2) -> pd.DataFrame:
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


def collect_market_dataset(pages: int = 6, sleep_seconds: float = 1.2) -> pd.DataFrame:
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


GRAPH_PALETTE = [
    "#61f0d1",
    "#ffb86b",
    "#7ab8ff",
    "#ff7f7f",
    "#9cff6b",
    "#ffd86b",
    "#cfa8ff",
    "#6be0ff",
]


def _bundle_color(index: int) -> str:
    return GRAPH_PALETTE[index % len(GRAPH_PALETTE)]


def build_graph_payload(
    bundles: list[dict[str, Any]],
    eligible_bundle_df: pd.DataFrame,
) -> dict[str, Any]:
    if eligible_bundle_df.empty:
        return {"nodes": [], "edges": [], "bundles": [], "summary": {"node_count": 0, "edge_count": 0}}

    bundle_lookup = {bundle["id"]: bundle for bundle in bundles}
    ordered_bundle_ids = [bundle["id"] for bundle in bundles if bundle["id"] in set(eligible_bundle_df["bundle_id"])]

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    node_lookup: dict[str, dict[str, Any]] = {}
    edge_keys: set[tuple[str, str, str, str | None]] = set()
    bundle_meta: list[dict[str, Any]] = []

    candidate_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    peer_lookup: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for bundle in bundles:
        for rank, candidate in enumerate(bundle["candidates"], start=1):
            candidate_lookup[(bundle["id"], candidate["id"])] = {
                "candidate": candidate,
                "rank": rank,
            }
            for peer in candidate["peer_assets"]:
                peer_lookup.setdefault((bundle["id"], peer["id"]), []).append(
                    {
                        "id": candidate["id"],
                        "name": candidate["name"],
                        "bundle_id": bundle["id"],
                        "bundle_label": bundle["label"],
                    }
                )

    type_priority = {
        "bundle": 3,
        "candidate": 2,
        "anchor_peer": 1,
        "eligible_asset": 0,
    }

    def register_node(node: dict[str, Any]) -> dict[str, Any]:
        existing = node_lookup.get(node["id"])
        if not existing:
            node_lookup[node["id"]] = node
            nodes.append(node)
            return node

        existing_bundle_ids = set(existing.get("bundle_ids", []))
        existing_bundle_labels = set(existing.get("bundle_labels", []))
        incoming_bundle_ids = set(node.get("bundle_ids", []))
        incoming_bundle_labels = set(node.get("bundle_labels", []))
        existing["bundle_ids"] = sorted(existing_bundle_ids | incoming_bundle_ids)
        existing["bundle_labels"] = sorted(existing_bundle_labels | incoming_bundle_labels)

        if type_priority.get(node.get("type"), -1) > type_priority.get(existing.get("type"), -1):
            existing["type"] = node["type"]

        if node.get("candidate"):
            existing["candidate"] = True
            existing["rank"] = node.get("rank")
            existing["gap_ratio"] = node.get("gap_ratio")
            existing["explanation"] = node.get("explanation")
            existing["matched_features"] = node.get("matched_features", [])
            existing["peer_assets"] = node.get("peer_assets", [])
        existing["market_cap"] = max(existing.get("market_cap") or 0, node.get("market_cap") or 0) or None
        existing["is_anchor_peer"] = bool(existing.get("is_anchor_peer")) or bool(node.get("is_anchor_peer"))
        existing["is_eligible"] = bool(existing.get("is_eligible", False)) or bool(node.get("is_eligible", False))

        related_candidates = existing.get("related_candidates", []) + node.get("related_candidates", [])
        if related_candidates:
            deduped = {(item["id"], item["bundle_id"]): item for item in related_candidates}
            existing["related_candidates"] = list(deduped.values())

        for axis in ("x", "y", "z"):
            if axis in node:
                existing[axis] = round((float(existing.get(axis, 0)) + float(node[axis])) / 2, 2)
        return existing

    def register_edge(edge: dict[str, Any]) -> None:
        edge_key = (edge["source"], edge["target"], edge["type"], edge.get("bundle_id"))
        if edge_key in edge_keys:
            return
        edge_keys.add(edge_key)
        edges.append(edge)

    bundle_count = max(len(ordered_bundle_ids), 1)
    for bundle_index, bundle_id in enumerate(ordered_bundle_ids):
        bundle = bundle_lookup[bundle_id]
        group = eligible_bundle_df[eligible_bundle_df["bundle_id"] == bundle_id].sort_values("market_cap", ascending=False, na_position="last").reset_index(drop=True)
        color = _bundle_color(bundle_index)
        angle = (2 * math.pi * bundle_index) / bundle_count
        hub_x = round(math.cos(angle) * 520, 2)
        hub_z = round(math.sin(angle) * 520, 2)
        hub_y = float(((bundle_index % 4) - 1.5) * 90)
        bundle_node_id = f"bundle:{bundle_id}"

        register_node(
            {
                "id": bundle_node_id,
                "type": "bundle",
                "label": bundle["label"],
                "bundle_id": bundle_id,
                "bundle_ids": [bundle_id],
                "bundle_labels": [bundle["label"]],
                "candidate_count": bundle["candidate_count"],
                "eligible_asset_count": int(len(group)),
                "top_gap_ratio": bundle["top_gap_ratio"],
                "description": bundle["description"],
                "top_candidates": [candidate["name"] for candidate in bundle["candidates"][:4]],
                "color": color,
                "x": hub_x,
                "y": hub_y,
                "z": hub_z,
            }
        )
        bundle_meta.append(
            {
                "id": bundle_id,
                "label": bundle["label"],
                "color": color,
                "candidate_count": bundle["candidate_count"],
                "eligible_asset_count": int(len(group)),
                "top_gap_ratio": bundle["top_gap_ratio"],
                "has_candidates": bundle["candidate_count"] > 0,
            }
        )

        asset_count = max(len(group), 1)
        for asset_index, asset in enumerate(group.itertuples(index=False)):
            asset_key = (bundle_id, asset.asset_id)
            candidate_meta = candidate_lookup.get(asset_key)
            related_candidates = peer_lookup.get(asset_key, [])
            if candidate_meta:
                node_type = "candidate"
                orbit_radius = 180 + 28 * (asset_index % 4)
                y_offset = ((asset_index % 5) - 2) * 34
            elif related_candidates:
                node_type = "anchor_peer"
                orbit_radius = 320 + 22 * (asset_index % 4)
                y_offset = ((asset_index % 4) - 1.5) * 54
            else:
                node_type = "eligible_asset"
                orbit_radius = 255 + 18 * (asset_index % 5)
                y_offset = ((asset_index % 6) - 2.5) * 28

            asset_angle = angle + (2 * math.pi * asset_index / asset_count)
            asset_x = hub_x + math.cos(asset_angle) * orbit_radius
            asset_z = hub_z + math.sin(asset_angle) * orbit_radius
            asset_y = hub_y + y_offset
            node = {
                "id": f"asset:{asset.asset_id}",
                "asset_id": asset.asset_id,
                "type": node_type,
                "candidate": bool(candidate_meta),
                "is_anchor_peer": bool(related_candidates),
                "is_eligible": True,
                "label": asset.name,
                "symbol": asset.symbol.upper(),
                "bundle_ids": [bundle_id],
                "bundle_labels": [bundle["label"]],
                "market_cap": float(asset.market_cap) if pd.notna(asset.market_cap) else None,
                "color": color,
                "x": round(asset_x, 2),
                "y": round(asset_y, 2),
                "z": round(asset_z, 2),
            }
            if candidate_meta:
                candidate = candidate_meta["candidate"]
                node.update(
                    {
                        "rank": candidate_meta["rank"],
                        "gap_ratio": candidate["gap_ratio"],
                        "matched_features": candidate["matched_features"],
                        "peer_assets": candidate["peer_assets"],
                        "explanation": candidate["explanation"],
                    }
                )
            if related_candidates:
                node["related_candidates"] = related_candidates

            register_node(node)
            register_edge(
                {
                    "source": bundle_node_id,
                    "target": f"asset:{asset.asset_id}",
                    "type": "bundle_membership",
                    "bundle_id": bundle_id,
                    "weight": candidate_meta["candidate"]["gap_ratio"] if candidate_meta else 0.35,
                    "color": color,
                }
            )

        for candidate in bundle["candidates"]:
            candidate_node_id = f"asset:{candidate['id']}"
            for peer_index, peer in enumerate(candidate["peer_assets"]):
                register_edge(
                    {
                        "source": candidate_node_id,
                        "target": f"asset:{peer['id']}",
                        "type": "peer_justification",
                        "bundle_id": bundle_id,
                        "weight": round(1 / (peer_index + 1), 3),
                        "color": color,
                    }
                )

    for node in nodes:
        if "bundle_ids" in node:
            node["bundle_ids"] = sorted(node["bundle_ids"])
        if "bundle_labels" in node:
            node["bundle_labels"] = sorted(node["bundle_labels"])
        if node.get("related_candidates"):
            node["related_candidates"] = sorted(node["related_candidates"], key=lambda item: (item["bundle_label"], item["name"]))

    focused_node_count = sum(1 for node in nodes if node["type"] != "eligible_asset" and (node["type"] != "bundle" or node.get("candidate_count", 0) > 0))
    eligible_asset_count = sum(1 for node in nodes if node.get("is_eligible"))
    return {
        "bundles": bundle_meta,
        "nodes": nodes,
        "edges": edges,
        "summary": {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "focused_node_count": focused_node_count,
            "eligible_asset_count": eligible_asset_count,
        },
    }


def build_screener_payload(raw_df: pd.DataFrame, config: FilterConfig | None = None) -> dict[str, Any]:
    active_config = config or FilterConfig()
    prepared_df = prepare_asset_frame(raw_df, active_config)
    bundles = build_bundle_payload(prepared_df, active_config)
    bundle_df = expand_bundle_memberships(prepared_df)
    eligible_bundle_df = bundle_df[bundle_df["passes_basic_filters"]].copy()
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
        "graph": build_graph_payload(bundles, eligible_bundle_df),
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
    html = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Crypto Similarity Graph</title>
  <style>
    :root {
      --bg: #04101c;
      --panel: rgba(7, 18, 32, 0.84);
      --line: rgba(142, 187, 255, 0.18);
      --text: #eef5ff;
      --muted: #99afc9;
      --accent: #61f0d1;
      --warm: #ffb86b;
      --shadow: 0 26px 80px rgba(0, 0, 0, 0.38);
      --radius: 24px;
    }

    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: "Avenir Next", "Segoe UI", sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(97, 240, 209, 0.14), transparent 28%),
        radial-gradient(circle at top right, rgba(255, 184, 107, 0.16), transparent 24%),
        linear-gradient(180deg, #020912 0%, #07111d 42%, #06111e 100%);
    }

    h1, h2, h3 {
      margin: 0;
      font-family: "Iowan Old Style", "Palatino Linotype", serif;
      letter-spacing: 0.01em;
    }

    .shell {
      width: min(1520px, calc(100vw - 28px));
      margin: 16px auto 32px;
      display: grid;
      gap: 16px;
    }

    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      backdrop-filter: blur(12px);
    }

    .hero {
      padding: 26px 28px;
      background: linear-gradient(135deg, rgba(8, 20, 35, 0.96), rgba(5, 14, 24, 0.9));
      position: relative;
      overflow: hidden;
    }

    .hero::after {
      content: "";
      position: absolute;
      width: 320px;
      height: 320px;
      right: -60px;
      top: -120px;
      border-radius: 50%;
      background: radial-gradient(circle, rgba(97, 240, 209, 0.16), transparent 68%);
      pointer-events: none;
    }

    .eyebrow {
      text-transform: uppercase;
      letter-spacing: 0.22em;
      font-size: 12px;
      color: var(--accent);
      margin-bottom: 10px;
    }

    .hero-copy {
      display: grid;
      grid-template-columns: minmax(0, 1.35fr) minmax(260px, 0.65fr);
      gap: 18px;
      align-items: end;
    }

    .hero p {
      margin: 12px 0 0;
      color: var(--muted);
      max-width: 860px;
      line-height: 1.6;
    }

    .stat-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-top: 22px;
    }

    .stat {
      padding: 16px 18px;
      border-radius: 20px;
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid rgba(142, 187, 255, 0.12);
    }

    .stat-label {
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }

    .stat-value {
      margin-top: 8px;
      font-size: 28px;
    }

    .toolbar {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
    }

    .control { padding: 16px 18px; }
    .control label { display: block; margin-bottom: 10px; color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; }
    .control select, .control button {
      width: 100%;
      border-radius: 16px;
      border: 1px solid rgba(142, 187, 255, 0.18);
      background: rgba(5, 13, 23, 0.86);
      color: var(--text);
      padding: 11px 13px;
      font: inherit;
      cursor: pointer;
    }

    .layout {
      display: grid;
      grid-template-columns: 280px minmax(0, 1fr) 320px;
      gap: 16px;
      min-height: 720px;
      align-items: stretch;
    }

    .sidebar, .detail { padding: 18px; }
    .sidebar h2, .detail h2 { font-size: 24px; margin-bottom: 14px; }
    .bundle-list { display: grid; gap: 10px; }
    .bundle-chip {
      width: 100%;
      border: 1px solid transparent;
      border-radius: 18px;
      background: rgba(255, 255, 255, 0.03);
      color: inherit;
      padding: 14px;
      text-align: left;
      cursor: pointer;
      transition: 140ms ease;
    }

    .bundle-chip:hover, .bundle-chip.active {
      border-color: rgba(97, 240, 209, 0.34);
      background: rgba(97, 240, 209, 0.08);
      transform: translateY(-1px);
    }

    .bundle-chip .meta { color: var(--muted); font-size: 13px; margin-top: 6px; }
    .bundle-swatch { width: 12px; height: 12px; border-radius: 999px; display: inline-block; margin-right: 8px; vertical-align: middle; }

    .stage {
      position: relative;
      overflow: hidden;
      min-height: 720px;
      background:
        radial-gradient(circle at 50% 12%, rgba(97, 240, 209, 0.06), transparent 28%),
        linear-gradient(180deg, rgba(6, 17, 30, 0.98), rgba(4, 11, 20, 0.94));
    }

    .stage canvas { width: 100%; height: 100%; display: block; }
    .hud {
      position: absolute;
      top: 16px;
      left: 16px;
      right: 16px;
      display: flex;
      justify-content: space-between;
      gap: 10px;
      pointer-events: none;
    }

    .hud-card {
      pointer-events: auto;
      padding: 10px 12px;
      border-radius: 16px;
      background: rgba(7, 18, 32, 0.72);
      border: 1px solid rgba(142, 187, 255, 0.14);
      color: var(--muted);
      font-size: 13px;
    }

    .detail p, .detail li { color: var(--muted); line-height: 1.55; }
    .detail ul { margin: 0; padding-left: 18px; }
    .metric-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; margin: 14px 0; }
    .metric { padding: 12px; border-radius: 16px; background: rgba(255,255,255,0.04); border: 1px solid rgba(142, 187, 255, 0.12); }
    .metric-label { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; }
    .metric-value { margin-top: 6px; font-size: 18px; }
    .pill-row { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
    .pill { padding: 7px 10px; border-radius: 999px; background: rgba(255,255,255,0.05); border: 1px solid rgba(142, 187, 255, 0.14); color: var(--muted); font-size: 12px; }
    .note { color: var(--muted); font-size: 13px; }

    @media (max-width: 1240px) {
      .layout { grid-template-columns: 1fr; }
      .hero-copy { grid-template-columns: 1fr; }
      .toolbar { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .stage { min-height: 620px; }
    }

    @media (max-width: 720px) {
      .shell { width: min(100vw - 18px, 100%); margin: 10px auto 20px; }
      .hero, .control, .sidebar, .detail { padding: 18px; }
      .toolbar, .stat-grid { grid-template-columns: 1fr; }
      .stage { min-height: 480px; }
      .hud { flex-direction: column; align-items: flex-start; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <section class="panel hero">
      <div class="eyebrow">Crypto Similarity Graph</div>
      <div class="hero-copy">
        <div>
          <h1>See the bundle structure instead of reading it as a table.</h1>
          <p>This graph centers bundle hubs, candidate nodes, and the larger peers that justified each pick. Drag to orbit, use the wheel to zoom, and click nodes to inspect why a coin is where it is.</p>
        </div>
        <div class="note">Primary view: 3D bundle map. Default density: surfaced candidates plus anchor peers only.</div>
      </div>
      <div class="stat-grid" id="stats"></div>
    </section>

    <section class="toolbar">
      <div class="panel control">
        <label for="bundleFilter">Bundle Focus</label>
        <select id="bundleFilter"></select>
      </div>
      <div class="panel control">
        <label for="nodeScope">Node Scope</label>
        <select id="nodeScope">
          <option value="focused">Candidates + Anchor Peers</option>
          <option value="eligible">All Eligible Assets</option>
        </select>
      </div>
      <div class="panel control">
        <label for="labelDensity">Labels</label>
        <select id="labelDensity">
          <option value="smart">Smart Labels</option>
          <option value="all">Show More</option>
          <option value="minimal">Minimal</option>
        </select>
      </div>
      <div class="panel control">
        <label>&nbsp;</label>
        <button id="resetView">Reset View</button>
      </div>
    </section>

    <section class="layout">
      <aside class="panel sidebar">
        <h2>Bundle Clusters</h2>
        <div class="bundle-list" id="bundleLegend"></div>
      </aside>

      <main class="panel stage">
        <canvas id="graphCanvas"></canvas>
        <div class="hud">
          <div class="hud-card">Drag: orbit • Wheel: zoom • Click: inspect</div>
          <div class="hud-card" id="hudSummary"></div>
        </div>
      </main>

      <aside class="panel detail" id="detailPanel"></aside>
    </section>
  </div>

  <script>
    const PAYLOAD = __PAYLOAD_JSON__;
    const GRAPH = PAYLOAD.graph;
    const NODE_MAP = new Map(GRAPH.nodes.map((node) => [node.id, node]));
    const canvas = document.getElementById('graphCanvas');
    const ctx = canvas.getContext('2d');

    const state = {
      selectedBundle: 'all',
      nodeScope: 'focused',
      labelDensity: 'smart',
      selectedId: GRAPH.nodes.find((node) => node.type === 'candidate')?.id || GRAPH.nodes[0]?.id || null,
      zoom: 1.08,
      rotX: -0.42,
      rotY: 0.56,
    };

    let projectedNodes = [];
    let isDragging = false;
    let dragMoved = false;
    let lastPointer = null;

    const formatCompact = (value) => new Intl.NumberFormat('en-US', { notation: 'compact', maximumFractionDigits: 1 }).format(value || 0);
    const formatCurrency = (value) => new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: value >= 1 ? 0 : 4 }).format(value || 0);
    const escapeHtml = (value) => String(value)
      .replaceAll('&', '&amp;')
      .replaceAll('<', '&lt;')
      .replaceAll('>', '&gt;')
      .replaceAll('"', '&quot;')
      .replaceAll("'", '&#039;');

    function resizeCanvas() {
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    function getVisibleNodes() {
      return GRAPH.nodes.filter((node) => {
        const inBundle = state.selectedBundle === 'all' || node.bundle_id === state.selectedBundle || (node.bundle_ids || []).includes(state.selectedBundle);
        if (!inBundle) return false;
        if (state.nodeScope === 'focused') {
          if (node.type === 'eligible_asset') return false;
          if (node.type === 'bundle' && !(node.candidate_count > 0)) return false;
        }
        return true;
      });
    }

    function getVisibleEdges(visibleIds) {
      return GRAPH.edges.filter((edge) => {
        return visibleIds.has(edge.source) && visibleIds.has(edge.target);
      });
    }

    function ensureSelection(visibleNodes) {
      if (!visibleNodes.length) {
        state.selectedId = null;
        return;
      }
      const visibleIds = new Set(visibleNodes.map((node) => node.id));
      if (!visibleIds.has(state.selectedId)) {
        state.selectedId = visibleNodes.find((node) => node.type === 'candidate')?.id || visibleNodes[0].id;
      }
    }

    function transformNode(node, width, height) {
      const cosY = Math.cos(state.rotY);
      const sinY = Math.sin(state.rotY);
      const cosX = Math.cos(state.rotX);
      const sinX = Math.sin(state.rotX);

      const x1 = node.x * cosY - node.z * sinY;
      const z1 = node.x * sinY + node.z * cosY;
      const y1 = node.y * cosX - z1 * sinX;
      const z2 = node.y * sinX + z1 * cosX;
      const camera = 1380 / state.zoom;
      const perspective = camera / (camera - z2);
      const screenX = width / 2 + x1 * perspective;
      const screenY = height / 2 + y1 * perspective;
      const marketCap = node.market_cap || 0;
      const capScale = marketCap ? Math.max(0, Math.log10(marketCap) - 7) : 0;
      const baseRadius = node.type === 'bundle' ? 18 : node.type === 'candidate' ? 10 : 7.5;

      return {
        ...node,
        depth: z2,
        screenX,
        screenY,
        radius: baseRadius + capScale * (node.type === 'bundle' ? 0.4 : 0.7),
      };
    }

    function shouldLabel(node) {
      if (state.labelDensity === 'minimal') return node.id === state.selectedId;
      if (state.labelDensity === 'all') return true;
      return node.type === 'bundle' || node.type === 'candidate' || node.id === state.selectedId;
    }

    function renderStats() {
      const visibleNodeTarget = state.nodeScope === 'focused' ? GRAPH.summary.focused_node_count : GRAPH.summary.node_count;
      const stats = [
        ['Assets Loaded', formatCompact(PAYLOAD.summary.total_assets)],
        ['Eligible Assets', formatCompact(PAYLOAD.summary.eligible_assets)],
        ['Visible Universe', formatCompact(visibleNodeTarget)],
        ['Graph Edges', formatCompact(GRAPH.summary.edge_count)],
      ];
      document.getElementById('stats').innerHTML = stats.map(([label, value]) => `
        <div class="stat">
          <div class="stat-label">${label}</div>
          <div class="stat-value">${value}</div>
        </div>
      `).join('');
    }

    function renderBundleFilter() {
      const select = document.getElementById('bundleFilter');
      select.innerHTML = '<option value="all">All Bundles</option>' + GRAPH.bundles.map((bundle) => `
        <option value="${bundle.id}">${escapeHtml(bundle.label)} (${bundle.candidate_count})</option>
      `).join('');
      select.value = state.selectedBundle;
    }

    function renderLegend() {
      document.getElementById('bundleLegend').innerHTML = GRAPH.bundles.map((bundle) => `
        <button class="bundle-chip ${bundle.id === state.selectedBundle ? 'active' : ''}" data-bundle="${bundle.id}">
          <div><span class="bundle-swatch" style="background:${bundle.color};"></span>${escapeHtml(bundle.label)}</div>
          <div class="meta">${bundle.candidate_count} candidates • ${bundle.eligible_asset_count} eligible</div>
          <div class="meta">Top peer gap: ${bundle.top_gap_ratio ? `${bundle.top_gap_ratio.toFixed(1)}x` : 'n/a'}</div>
        </button>
      `).join('');

      document.querySelectorAll('[data-bundle]').forEach((button) => {
        button.addEventListener('click', () => {
          state.selectedBundle = button.dataset.bundle;
          document.getElementById('bundleFilter').value = state.selectedBundle;
          draw();
        });
      });
    }

    function renderDetail(selectedNode) {
      const detail = document.getElementById('detailPanel');
      if (!selectedNode) {
        detail.innerHTML = '<h2>No Selection</h2><p>Select a node to inspect its bundle, ranking, and peer relationships.</p>';
        return;
      }

      if (selectedNode.type === 'bundle') {
        detail.innerHTML = `
          <h2>${escapeHtml(selectedNode.label)}</h2>
          <p>${escapeHtml(selectedNode.description || 'Bundle hub')}</p>
          <div class="metric-grid">
            <div class="metric"><div class="metric-label">Candidates</div><div class="metric-value">${selectedNode.candidate_count}</div></div>
            <div class="metric"><div class="metric-label">Eligible Assets</div><div class="metric-value">${selectedNode.eligible_asset_count}</div></div>
            <div class="metric"><div class="metric-label">Top Gap</div><div class="metric-value">${selectedNode.top_gap_ratio ? `${selectedNode.top_gap_ratio.toFixed(1)}x` : 'n/a'}</div></div>
            <div class="metric"><div class="metric-label">Node Type</div><div class="metric-value">Bundle Hub</div></div>
          </div>
          <div class="pill-row">${(selectedNode.top_candidates || []).map((name) => `<span class="pill">${escapeHtml(name)}</span>`).join('')}</div>
          <p class="note">Bundle hubs anchor each cluster and keep the layout stable while you orbit the graph.</p>
        `;
        return;
      }

      if (selectedNode.type === 'candidate') {
        detail.innerHTML = `
          <h2>${escapeHtml(selectedNode.label)}</h2>
          <p>${escapeHtml(selectedNode.explanation || 'Candidate node')}</p>
          <div class="metric-grid">
            <div class="metric"><div class="metric-label">Symbol</div><div class="metric-value">${escapeHtml(selectedNode.symbol || '')}</div></div>
            <div class="metric"><div class="metric-label">Rank</div><div class="metric-value">${selectedNode.rank || 'n/a'}</div></div>
            <div class="metric"><div class="metric-label">Peer Gap</div><div class="metric-value">${selectedNode.gap_ratio ? `${selectedNode.gap_ratio.toFixed(1)}x` : 'n/a'}</div></div>
            <div class="metric"><div class="metric-label">Market Cap</div><div class="metric-value">${formatCurrency(selectedNode.market_cap)}</div></div>
          </div>
          <div class="pill-row">${(selectedNode.bundle_labels || []).map((label) => `<span class="pill">${escapeHtml(label)}</span>`).join('')}</div>
          <div class="pill-row">${(selectedNode.matched_features || []).map((feature) => `<span class="pill">${escapeHtml(feature)}</span>`).join('')}</div>
          <h3 style="margin-top:18px;">Anchor Peers</h3>
          <ul>${(selectedNode.peer_assets || []).map((peer) => `<li>${escapeHtml(peer.name)} • ${formatCompact(peer.market_cap)}</li>`).join('')}</ul>
        `;
        return;
      }

      if (selectedNode.type === 'eligible_asset') {
        detail.innerHTML = `
          <h2>${escapeHtml(selectedNode.label)}</h2>
          <p>Eligible asset node. This coin passed the screener filters for at least one bundle, but it is not currently surfaced as a candidate or used as an anchor peer in the focused view.</p>
          <div class="metric-grid">
            <div class="metric"><div class="metric-label">Symbol</div><div class="metric-value">${escapeHtml(selectedNode.symbol || '')}</div></div>
            <div class="metric"><div class="metric-label">Market Cap</div><div class="metric-value">${formatCurrency(selectedNode.market_cap)}</div></div>
            <div class="metric"><div class="metric-label">Bundles</div><div class="metric-value">${(selectedNode.bundle_labels || []).length}</div></div>
            <div class="metric"><div class="metric-label">Node Type</div><div class="metric-value">Eligible Asset</div></div>
          </div>
          <div class="pill-row">${(selectedNode.bundle_labels || []).map((label) => `<span class="pill">${escapeHtml(label)}</span>`).join('')}</div>
        `;
        return;
      }

      detail.innerHTML = `
        <h2>${escapeHtml(selectedNode.label)}</h2>
        <p>Anchor peer node. These are the larger reference assets that justify a candidate’s placement inside the same bundle.</p>
        <div class="metric-grid">
          <div class="metric"><div class="metric-label">Symbol</div><div class="metric-value">${escapeHtml(selectedNode.symbol || '')}</div></div>
          <div class="metric"><div class="metric-label">Market Cap</div><div class="metric-value">${formatCurrency(selectedNode.market_cap)}</div></div>
          <div class="metric"><div class="metric-label">Bundles</div><div class="metric-value">${(selectedNode.bundle_labels || []).length}</div></div>
          <div class="metric"><div class="metric-label">Referenced By</div><div class="metric-value">${(selectedNode.related_candidates || []).length}</div></div>
        </div>
        <div class="pill-row">${(selectedNode.bundle_labels || []).map((label) => `<span class="pill">${escapeHtml(label)}</span>`).join('')}</div>
        <h3 style="margin-top:18px;">Connected Candidates</h3>
        <ul>${(selectedNode.related_candidates || []).map((item) => `<li>${escapeHtml(item.name)} • ${escapeHtml(item.bundle_label)}</li>`).join('')}</ul>
      `;
    }

    function draw() {
      renderStats();
      resizeCanvas();
      const width = canvas.clientWidth;
      const height = canvas.clientHeight;
      ctx.clearRect(0, 0, width, height);

      const visibleNodes = getVisibleNodes();
      ensureSelection(visibleNodes);
      const visibleIds = new Set(visibleNodes.map((node) => node.id));
      const visibleEdges = getVisibleEdges(visibleIds);
      projectedNodes = visibleNodes.map((node) => transformNode(node, width, height));
      const projectedMap = new Map(projectedNodes.map((node) => [node.id, node]));

      const edgeAlpha = state.nodeScope === 'focused' ? 0.2 : 0.16;
      visibleEdges
        .map((edge) => ({ edge, from: projectedMap.get(edge.source), to: projectedMap.get(edge.target) }))
        .filter((item) => item.from && item.to)
        .sort((left, right) => ((left.from.depth + left.to.depth) / 2) - ((right.from.depth + right.to.depth) / 2))
        .forEach(({ edge, from, to }) => {
          ctx.save();
          ctx.globalAlpha = edgeAlpha;
          ctx.lineWidth = edge.type === 'bundle_membership' ? 1.4 : 1;
          ctx.strokeStyle = edge.color;
          ctx.beginPath();
          ctx.moveTo(from.screenX, from.screenY);
          ctx.lineTo(to.screenX, to.screenY);
          ctx.stroke();
          ctx.restore();
        });

      projectedNodes
        .slice()
        .sort((left, right) => left.depth - right.depth)
        .forEach((node) => {
          ctx.save();
          ctx.beginPath();
          ctx.fillStyle = node.color || '#61f0d1';
          ctx.shadowBlur = node.type === 'bundle' ? 18 : 12;
          ctx.shadowColor = node.color || '#61f0d1';
          ctx.arc(node.screenX, node.screenY, node.radius, 0, Math.PI * 2);
          ctx.fill();
          ctx.shadowBlur = 0;
          if (node.id === state.selectedId) {
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.arc(node.screenX, node.screenY, node.radius + 5, 0, Math.PI * 2);
            ctx.stroke();
          }
          if (node.type === 'bundle') {
            ctx.strokeStyle = 'rgba(255,255,255,0.45)';
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.arc(node.screenX, node.screenY, node.radius + 8, 0, Math.PI * 2);
            ctx.stroke();
          }
          if (shouldLabel(node)) {
            ctx.font = node.type === 'bundle' ? '600 14px Avenir Next' : '500 12px Avenir Next';
            ctx.fillStyle = '#eef5ff';
            ctx.fillText(node.label, node.screenX + node.radius + 8, node.screenY - 6);
          }
          ctx.restore();
        });

      document.getElementById('hudSummary').textContent = `${visibleNodes.length} visible nodes • ${visibleEdges.length} visible edges`;
      renderLegend();
      renderDetail(projectedNodes.find((node) => node.id === state.selectedId) || null);
    }

    function pickNode(clientX, clientY) {
      const rect = canvas.getBoundingClientRect();
      const x = clientX - rect.left;
      const y = clientY - rect.top;
      let best = null;
      let bestDistance = Infinity;
      projectedNodes.forEach((node) => {
        const dx = node.screenX - x;
        const dy = node.screenY - y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        if (distance <= node.radius + 8 && distance < bestDistance) {
          bestDistance = distance;
          best = node;
        }
      });
      return best;
    }

    canvas.addEventListener('pointerdown', (event) => {
      isDragging = true;
      dragMoved = false;
      lastPointer = { x: event.clientX, y: event.clientY };
      canvas.setPointerCapture(event.pointerId);
    });

    canvas.addEventListener('pointermove', (event) => {
      if (!isDragging || !lastPointer) return;
      const dx = event.clientX - lastPointer.x;
      const dy = event.clientY - lastPointer.y;
      if (Math.abs(dx) > 1 || Math.abs(dy) > 1) dragMoved = true;
      state.rotY += dx * 0.005;
      state.rotX = Math.max(-1.1, Math.min(1.1, state.rotX + dy * 0.004));
      lastPointer = { x: event.clientX, y: event.clientY };
      draw();
    });

    canvas.addEventListener('pointerup', (event) => {
      if (!dragMoved) {
        const picked = pickNode(event.clientX, event.clientY);
        if (picked) {
          state.selectedId = picked.id;
          draw();
        }
      }
      isDragging = false;
      lastPointer = null;
      canvas.releasePointerCapture(event.pointerId);
    });

    canvas.addEventListener('wheel', (event) => {
      event.preventDefault();
      state.zoom = Math.max(0.55, Math.min(2.1, state.zoom + Math.sign(event.deltaY) * 0.08));
      draw();
    }, { passive: false });

    document.getElementById('bundleFilter').addEventListener('change', (event) => {
      state.selectedBundle = event.target.value;
      draw();
    });
    document.getElementById('nodeScope').addEventListener('change', (event) => {
      state.nodeScope = event.target.value;
      draw();
    });
    document.getElementById('labelDensity').addEventListener('change', (event) => {
      state.labelDensity = event.target.value;
      draw();
    });
    document.getElementById('resetView').addEventListener('click', () => {
      state.zoom = 1.08;
      state.rotX = -0.42;
      state.rotY = 0.56;
      draw();
    });

    window.addEventListener('resize', draw);

    renderStats();
    renderBundleFilter();
    draw();
  </script>
</body>
</html>
"""
    return html.replace("__PAYLOAD_JSON__", payload_json)


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
