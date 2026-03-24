import unittest
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from screener import (
    FilterConfig,
    assign_bundles,
    build_screener_payload,
    build_snapshot_frame,
    build_validation_report,
    load_snapshot_history,
    parse_categories,
    save_snapshot_frame,
)


class ScreenerTests(unittest.TestCase):
    def test_parse_categories_handles_stringified_lists(self) -> None:
        parsed = parse_categories("['Stablecoins', 'USD Stablecoin']")
        self.assertEqual(parsed, ["Stablecoins", "USD Stablecoin"])

    def test_assign_bundles_supports_multiple_matches(self) -> None:
        bundles = assign_bundles(["Stablecoins", "Decentralized Finance (DeFi)"])
        self.assertEqual(set(bundles), {"stablecoins", "defi"})

    def test_build_payload_ranks_assets_by_peer_gap(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "id": "alpha",
                    "symbol": "alp",
                    "name": "Alpha",
                    "current_price": 12.0,
                    "market_cap": 12_000_000_000,
                    "total_volume": 3_000_000_000,
                    "price_change_percentage_24h": 2.1,
                    "categories": "['Layer 1 (L1)', 'Smart Contract Platform']",
                    "hashing_algorithm": "PoS",
                    "genesis_date": "2016-01-01",
                },
                {
                    "id": "beta",
                    "symbol": "bet",
                    "name": "Beta",
                    "current_price": 6.0,
                    "market_cap": 4_000_000_000,
                    "total_volume": 1_500_000_000,
                    "price_change_percentage_24h": 1.4,
                    "categories": "['Layer 1 (L1)', 'Smart Contract Platform']",
                    "hashing_algorithm": "PoS",
                    "genesis_date": "2017-01-01",
                },
                {
                    "id": "gamma",
                    "symbol": "gam",
                    "name": "Gamma",
                    "current_price": 1.5,
                    "market_cap": 1_000_000_000,
                    "total_volume": 250_000_000,
                    "price_change_percentage_24h": -0.8,
                    "categories": "['Layer 1 (L1)', 'Smart Contract Platform']",
                    "hashing_algorithm": "PoS",
                    "genesis_date": "2018-01-01",
                },
                {
                    "id": "delta",
                    "symbol": "del",
                    "name": "Delta",
                    "current_price": 0.4,
                    "market_cap": 200_000_000,
                    "total_volume": 75_000_000,
                    "price_change_percentage_24h": 4.3,
                    "categories": "['Layer 1 (L1)', 'Smart Contract Platform']",
                    "hashing_algorithm": "PoS",
                    "genesis_date": "2019-01-01",
                },
            ]
        )

        payload = build_screener_payload(
            df,
            FilterConfig(
                min_market_cap_usd=100_000,
                min_volume_usd=10_000,
                min_age_days=30,
                min_peer_count=2,
                max_peer_count=2,
                min_gap_ratio=1.2,
            ),
        )

        layer1 = next(bundle for bundle in payload["bundles"] if bundle["id"] == "layer1")
        self.assertGreaterEqual(layer1["candidate_count"], 2)
        self.assertEqual(layer1["candidates"][0]["id"], "delta")
        self.assertEqual(layer1["candidates"][1]["id"], "gamma")

    def test_build_payload_filters_out_young_or_illiquid_assets(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "id": "old-one",
                    "symbol": "old",
                    "name": "Old One",
                    "current_price": 2.0,
                    "market_cap": 800_000_000,
                    "total_volume": 50_000_000,
                    "price_change_percentage_24h": 1.1,
                    "categories": "['Stablecoins']",
                    "hashing_algorithm": "",
                    "genesis_date": "2019-01-01",
                },
                {
                    "id": "old-two",
                    "symbol": "od2",
                    "name": "Old Two",
                    "current_price": 1.0,
                    "market_cap": 400_000_000,
                    "total_volume": 35_000_000,
                    "price_change_percentage_24h": 0.3,
                    "categories": "['Stablecoins']",
                    "hashing_algorithm": "",
                    "genesis_date": "2018-01-01",
                },
                {
                    "id": "young-coin",
                    "symbol": "yng",
                    "name": "Young Coin",
                    "current_price": 0.8,
                    "market_cap": 250_000_000,
                    "total_volume": 25_000_000,
                    "price_change_percentage_24h": 3.5,
                    "categories": "['Stablecoins']",
                    "hashing_algorithm": "",
                    "genesis_date": "2026-01-15",
                },
                {
                    "id": "thin-coin",
                    "symbol": "thn",
                    "name": "Thin Coin",
                    "current_price": 0.5,
                    "market_cap": 120_000_000,
                    "total_volume": 500,
                    "price_change_percentage_24h": -1.4,
                    "categories": "['Stablecoins']",
                    "hashing_algorithm": "",
                    "genesis_date": "2017-01-01",
                },
            ]
        )

        payload = build_screener_payload(
            df,
            FilterConfig(
                min_market_cap_usd=100_000,
                min_volume_usd=10_000,
                min_age_days=180,
                min_peer_count=2,
                max_peer_count=2,
                min_gap_ratio=1.2,
            ),
        )

        stablecoins = next(bundle for bundle in payload["bundles"] if bundle["id"] == "stablecoins")
        candidate_ids = [candidate["id"] for candidate in stablecoins["candidates"]]
        self.assertNotIn("young-coin", candidate_ids)
        self.assertNotIn("thin-coin", candidate_ids)

    def test_build_snapshot_frame_marks_candidates_and_eligible_assets(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "id": "alpha",
                    "symbol": "alp",
                    "name": "Alpha",
                    "current_price": 12.0,
                    "market_cap": 12_000_000_000,
                    "total_volume": 3_000_000_000,
                    "price_change_percentage_24h": 2.1,
                    "categories": "['Layer 1 (L1)', 'Smart Contract Platform']",
                    "hashing_algorithm": "PoS",
                    "genesis_date": "2016-01-01",
                },
                {
                    "id": "beta",
                    "symbol": "bet",
                    "name": "Beta",
                    "current_price": 6.0,
                    "market_cap": 4_000_000_000,
                    "total_volume": 1_500_000_000,
                    "price_change_percentage_24h": 1.4,
                    "categories": "['Layer 1 (L1)', 'Smart Contract Platform']",
                    "hashing_algorithm": "PoS",
                    "genesis_date": "2017-01-01",
                },
                {
                    "id": "gamma",
                    "symbol": "gam",
                    "name": "Gamma",
                    "current_price": 1.5,
                    "market_cap": 1_000_000_000,
                    "total_volume": 250_000_000,
                    "price_change_percentage_24h": -0.8,
                    "categories": "['Layer 1 (L1)', 'Smart Contract Platform']",
                    "hashing_algorithm": "PoS",
                    "genesis_date": "2018-01-01",
                },
                {
                    "id": "delta",
                    "symbol": "del",
                    "name": "Delta",
                    "current_price": 0.4,
                    "market_cap": 200_000_000,
                    "total_volume": 75_000_000,
                    "price_change_percentage_24h": 4.3,
                    "categories": "['Layer 1 (L1)', 'Smart Contract Platform']",
                    "hashing_algorithm": "PoS",
                    "genesis_date": "2019-01-01",
                },
            ]
        )

        snapshot = build_snapshot_frame(
            df,
            FilterConfig(
                min_market_cap_usd=100_000,
                min_volume_usd=10_000,
                min_age_days=30,
                min_peer_count=2,
                max_peer_count=2,
                min_gap_ratio=1.2,
            ),
            snapshot_date=date(2026, 3, 23),
        )

        self.assertIn("is_candidate", snapshot.columns)
        self.assertIn("rank", snapshot.columns)
        self.assertTrue((snapshot["snapshot_date"] == "2026-03-23").all())
        delta_row = snapshot[snapshot["asset_id"] == "delta"].iloc[0]
        self.assertTrue(bool(delta_row["is_candidate"]))
        self.assertEqual(int(delta_row["rank"]), 1)

    def test_validation_report_compares_candidates_to_bundle_median(self) -> None:
        day_zero = pd.DataFrame(
            [
                {
                    "snapshot_date": "2026-01-01",
                    "asset_id": "cand",
                    "symbol": "CND",
                    "name": "Candidate",
                    "bundle_id": "layer1",
                    "bundle_label": "Layer 1",
                    "is_candidate": True,
                    "rank": 1,
                    "gap_ratio": 2.5,
                    "current_price": 10.0,
                    "market_cap": 100_000_000,
                    "total_volume": 10_000_000,
                },
                {
                    "snapshot_date": "2026-01-01",
                    "asset_id": "peer-a",
                    "symbol": "PRA",
                    "name": "Peer A",
                    "bundle_id": "layer1",
                    "bundle_label": "Layer 1",
                    "is_candidate": False,
                    "rank": None,
                    "gap_ratio": None,
                    "current_price": 10.0,
                    "market_cap": 200_000_000,
                    "total_volume": 12_000_000,
                },
                {
                    "snapshot_date": "2026-01-01",
                    "asset_id": "peer-b",
                    "symbol": "PRB",
                    "name": "Peer B",
                    "bundle_id": "layer1",
                    "bundle_label": "Layer 1",
                    "is_candidate": False,
                    "rank": None,
                    "gap_ratio": None,
                    "current_price": 10.0,
                    "market_cap": 300_000_000,
                    "total_volume": 14_000_000,
                },
                {
                    "snapshot_date": "2026-01-01",
                    "asset_id": "peer-c",
                    "symbol": "PRC",
                    "name": "Peer C",
                    "bundle_id": "layer1",
                    "bundle_label": "Layer 1",
                    "is_candidate": False,
                    "rank": None,
                    "gap_ratio": None,
                    "current_price": 10.0,
                    "market_cap": 400_000_000,
                    "total_volume": 16_000_000,
                },
            ]
        )
        day_thirty = pd.DataFrame(
            [
                {
                    "snapshot_date": "2026-01-31",
                    "asset_id": "cand",
                    "symbol": "CND",
                    "name": "Candidate",
                    "bundle_id": "layer1",
                    "bundle_label": "Layer 1",
                    "is_candidate": True,
                    "rank": 1,
                    "gap_ratio": 2.5,
                    "current_price": 12.0,
                    "market_cap": 120_000_000,
                    "total_volume": 11_000_000,
                },
                {
                    "snapshot_date": "2026-01-31",
                    "asset_id": "peer-a",
                    "symbol": "PRA",
                    "name": "Peer A",
                    "bundle_id": "layer1",
                    "bundle_label": "Layer 1",
                    "is_candidate": False,
                    "rank": None,
                    "gap_ratio": None,
                    "current_price": 10.0,
                    "market_cap": 205_000_000,
                    "total_volume": 12_000_000,
                },
                {
                    "snapshot_date": "2026-01-31",
                    "asset_id": "peer-b",
                    "symbol": "PRB",
                    "name": "Peer B",
                    "bundle_id": "layer1",
                    "bundle_label": "Layer 1",
                    "is_candidate": False,
                    "rank": None,
                    "gap_ratio": None,
                    "current_price": 9.0,
                    "market_cap": 290_000_000,
                    "total_volume": 13_000_000,
                },
                {
                    "snapshot_date": "2026-01-31",
                    "asset_id": "peer-c",
                    "symbol": "PRC",
                    "name": "Peer C",
                    "bundle_id": "layer1",
                    "bundle_label": "Layer 1",
                    "is_candidate": False,
                    "rank": None,
                    "gap_ratio": None,
                    "current_price": 8.0,
                    "market_cap": 360_000_000,
                    "total_volume": 15_000_000,
                },
            ]
        )

        with TemporaryDirectory() as tmp_dir:
            snapshot_dir = Path(tmp_dir)
            save_snapshot_frame(day_zero, output_dir=snapshot_dir)
            save_snapshot_frame(day_thirty, output_dir=snapshot_dir)
            history = load_snapshot_history(snapshot_dir)
            report = build_validation_report(history, horizon_days=30, min_benchmark_assets=3)

        self.assertEqual(report["overall_summary"]["candidate_observations"], 1)
        self.assertEqual(report["overall_summary"]["hit_rate"], 1.0)
        detail = report["candidate_details"].iloc[0]
        self.assertAlmostEqual(detail["forward_return"], 0.2)
        self.assertAlmostEqual(detail["bundle_median_return"], -0.05)
        self.assertAlmostEqual(detail["excess_return"], 0.25)
        self.assertTrue(bool(detail["beat_bundle_median"]))


if __name__ == "__main__":
    unittest.main()
