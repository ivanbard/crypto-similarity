import unittest

import pandas as pd

from screener import FilterConfig, assign_bundles, build_screener_payload, parse_categories


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


if __name__ == "__main__":
    unittest.main()
