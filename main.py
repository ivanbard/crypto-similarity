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
    headers = {"x-messarir-api-key": MESSARI_KEY} if MESSARI_KEY else {}
    url = f"https://data.messari.io/api/v1/assets/{asset_slug}/metrics"

    response = requests.get(url, headers=headers)

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

def main():
    df=fetch_coingecko()
    #df = pd.DataFrame(all_data)
    print(df)
    #save to csv, so i have a proper dataset to work with rather than json bs

if __name__ == "__main__":
    main()