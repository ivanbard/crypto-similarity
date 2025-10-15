import os, requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
CG_KEY = os.getenv("COINGECKO-KEY")

url = "https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd"
headers = {"x-cg-demo-api-key": CG_KEY}

# print(f"{CG_KEY}")

response = requests.get(url, headers=headers)
data = response.json()

df = pd.DataFrame(data)
print(df)
df.to_csv('crypto.csv', index=False)
#save to csv, so i have a proper dataset to work with rather than json bs