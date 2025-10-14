import os, requests
from dotenv import load_dotenv

load_dotenv()
CG_KEY = os.getenv("COINGECKO-KEY")

url = "https://api.coingecko.com/api/v3/coins/markets"
headers = {"x-cg-demo-api-key": "{CG_KEY}"}

print(f"{CG_KEY}")