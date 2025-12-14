import os, requests, json, time
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

CG_KEY = os.getenv("COINGECKO-KEY")
MESSARI_KEY = os.getenv("MESSARI-KEY")

url = "https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&per_page=250&page={}"


def fetch_coingecko(pages=3):
    headers = {"x-cg-demo-api-key": CG_KEY}
    all_data = []

    for page in range(1, pages+1):
        url = f"https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&per_page=250&page={page}"
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            data = response.json()
            all_data.extend(data)
            print(f"fetched page {page}: {len(data)} coins")
        else:
            print("error fetching coingecko")

        # pause ever iteration to prevent hitting rate limit
        time.sleep(1) 
    return pd.DataFrame(all_data)

def fetch_coingecko_details(coin_id):
    headers = {"x-cg-demo-api-key": CG_KEY}
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}"
    
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        return {
            "id": coin_id,
            "categories": data.get("categories", []),
            "hashing_algorithm": data.get("hashing_algorithm"),
            "genesis_date": data.get("genesis_date"),
            "sentiment_up": data.get("sentiment_votes_up_percentage"),
            "sentiment_down": data.get("sentiment_votes_down_percentage"),
            "developer_score": data.get("developer_score"),
            "community_score": data.get("community_score"),
            "liquidity_score": data.get("liquidity_score"),
        }
    return None

def fetch_messari(asset_slug):
    #all_data = []
    headers = {"x-messari-api-key": MESSARI_KEY} if MESSARI_KEY else {}
    url = f"https://data.messari.io/api/v1/assets/{asset_slug}/metrics"

    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print(f"messari metrics error for {asset_slug}: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json().get("data", {})
        market = data.get("market_data", {})
        roi = data.get("roi_data", {})

        return {
            "id": asset_slug,
            "price_usd": market.get("price_usd"),
            "volume_24h": market.get("volume_last_24_hours"),
            "volatility_30d": market.get("volatility_last_30_days"),
            "percent_change_24h": market.get("percent_change_usd_last_24_hours"),
            "percent_change_7d": roi.get("percent_change_last_1_week"),
            "percent_change_30d": roi.get("percent_change_last_1_month"),
            "percent_change_90d": roi.get("percent_change_last_3_months"),
        }

    return None #if status != 200

def fetch_messari_prof(asset_slug):
    headers = {"x-messari-api-key": MESSARI_KEY} if MESSARI_KEY else {}
    url = f"https://data.messari.io/api/v1/assets/{asset_slug}/profile"
    
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print(f"messari profile error for {asset_slug}: {response.status_code}")

    if response.status_code == 200:
        data = response.json().get("data", {})
        profile = data.get("profile")

        general = profile.get("general", {})
        tech = profile.get("technology", {})
        econ = profile.get("economics", {})

        return {
            "id": asset_slug,
            "sector": general.get("overview", {}).get("sector"),
            "category": general.get("overview", {}).get("category"),
            "tag_line": general.get("overview", {}).get("tagline"),
            "token_type": general.get("overview", {}).get("token_type"),
            "consensus_mechanism": tech.get("overview", {}).get("consensus_mechanism"),
            "block_time": tech.get("overview", {}).get("block_time_interval"),
            "launch_style": econ.get("launch", {}).get("launch_style"),
            "initial_distribution": econ.get("launch", {}).get("initial_distribution"),
        }
    return None

def collect_all_data(num_coins=100):
    print("collecting coingecko market data\n")
    market_df = fetch_coingecko(pages=3)
    coin_ids = market_df["id"].head(num_coins).tolist()

    print("collecting coingecko coin details\n")
    cg_details = []
    for i, coin_id in enumerate(coin_ids):
        detail = fetch_coingecko_details(coin_id)
        if detail:
            cg_details.append(detail)
            print(f"  [{i+1}/{len(coin_ids)}] {coin_id}")
        time.sleep(1.5)
    
    cg_details_df = pd.DataFrame(cg_details)

    print("collecting messari profiles\n")
    messari_profiles = []
    for i, coin_id in enumerate(coin_ids):
        profile = fetch_messari_prof(coin_id)
        if profile:
            messari_profiles.append(profile)
            print(f"  [{i+1}/{len(coin_ids)}] {coin_id}")
        time.sleep(0.5)

    messari_profiles_df = pd.DataFrame(messari_profiles)

    print("collecting messari metrics\n")
    messari_metrics = []
    for i, coin_id in enumerate(coin_ids):
        metrics = fetch_messari(coin_id)
        if metrics:
            messari_metrics.append(metrics)
            print(f"  [{i+1}/{len(coin_ids)}] {coin_id}")
        time.sleep(0.5)

    messari_metrics_df = pd.DataFrame(messari_metrics)

    print("merging datasets\n")
    final_df = market_df[["id", "symbol", "name", "current_price", "market_cap", 
                          "total_volume", "price_change_percentage_24h"]].copy()
    
    # only merge if dfs are non empty and have 'id' col.
    if not cg_details_df.empty and "id" in cg_details_df.columns:
        final_df = final_df.merge(cg_details_df, on="id", how="left")
    
    if not messari_profiles_df.empty and "id" in messari_profiles_df.columns:
        final_df = final_df.merge(messari_profiles_df, on="id", how="left")
    else:
        print("Warning: No Messari profile data collected")
    
    if not messari_metrics_df.empty and "id" in messari_metrics_df.columns:
        final_df = final_df.merge(messari_metrics_df, on="id", how="left", suffixes=("", "_messari"))
    else:
        print("Warning: No Messari metrics data collected")
    
    return final_df

def main():
    df = collect_all_data(num_coins=100)
    
    #make dirs
    os.makedirs("data/raw", exist_ok=True)
    df.to_csv("data/raw/crypto_features_raw.csv", index=False)
    print(f"\nSaved {len(df)} coins to data/raw/crypto_features_raw.csv")
    
    print("\nColumns collected:")
    print(df.columns.tolist())
    print("\nSample data:")
    print(df.head())

if __name__ == "__main__":
    main()