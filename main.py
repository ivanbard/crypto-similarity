import os, requests, json
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
CG_KEY = os.getenv("COINGECKO-KEY")

url = "https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&per_page=250&page={}"
headers = {"x-cg-demo-api-key": CG_KEY}

# print(f"{CG_KEY}")

all_data = []
for page in range(1,3):
    url = f"https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&per_page=250&page={page}"
    response = requests.get(url, headers=headers)
    data = response.json()
    all_data.extend(data)
    print(f"fetched page {page}: {len(data)} coins")


df = pd.DataFrame(all_data)
print(df)
df.to_csv('crypto.csv', index=False)
#save to csv, so i have a proper dataset to work with rather than json bs