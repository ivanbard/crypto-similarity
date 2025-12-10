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



#for page in range(1,3):

#df = pd.DataFrame(all_data)
#print(df)
#save to csv, so i have a proper dataset to work with rather than json bs