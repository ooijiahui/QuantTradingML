import pandas as pd
import requests, time
from functools import reduce

# API Key
api_key = "r1zjrKRmvX6Y8XIDgJ8eQgOJ8vHCfFQ5BggC0H1Rom57YaJj"

# Function to fetch all data
def fetch_all_data():
    
    # List of endpoints and parameters
    calls = [
        #################### Miner Flows ####################
        # Inflows, Outflows
        {"endpoint": "/cryptoquant/btc/miner-flows/inflow", "params": {"miner": "all_miner", "window": "hour", "start_time": 1514764800000, "end_time": 1672487999000 }},
        {"endpoint": "/cryptoquant/btc/miner-flows/outflow", "params": {"miner": "all_miner", "window": "hour", "start_time": 1514764800000, "end_time": 1672487999000 }},
        #################### Exchange Flows  ####################
        # Inflows, Outflows
        {"endpoint": "/cryptoquant/btc/exchange-flows/inflow", "params": {"exchange": "all_exchange", "window": "hour", "start_time": 1514764800000, "end_time": 1672487999000 }},
        {"endpoint": "/cryptoquant/btc/exchange-flows/outflow", "params": {"exchange": "all_exchange", "window": "hour", "start_time": 1514764800000, "end_time": 1672487999000 }},
        #################### Network Data  ####################
        # Hashrate, Blockreward
        {"endpoint": "/cryptoquant/btc/network-data/hashrate", "params": {"window": "hour", "start_time": 1514764800000, "end_time": 1672487999000 }},
        {"endpoint": "/cryptoquant/btc/network-data/blockreward", "params": {"window": "hour", "start_time": 1514764800000, "end_time": 1672487999000 }},
         #################### Market Data  ####################
        # Price
        {"endpoint": "/cryptoquant/btc/market-data/price-ohlcv", "params": {"window": "hour", "start_time": 1514764800000, "end_time": 1672487999000 }},
        #################### Whale Data  ####################
        # Exchange Whale Ratio
        {"endpoint": "/cryptoquant/btc/flow-indicator/exchange-whale-ratio", "params": {"exchange": "all_exchange", "window": "hour", "start_time": 1514764800000, "end_time": 1672487999000 }},
    ]

    # Collect all DataFrames
    dfs = []
    for call in calls:
        df = fetch_data(endpoint=call["endpoint"], params=call["params"])
        dfs.append(df)

    # Combine all into one DataFrame
    combined_df = reduce(
        lambda left, right: pd.merge(
            left, right, left_index=True, right_index=True, how='outer', suffixes=('', '_dup')
        ),
        dfs
    )
    return combined_df

# Function to fetch data from an endpoint
def fetch_data(endpoint: str, params: dict) -> pd.DataFrame:
    url = f"https://api.datasource.cybotrade.rs{endpoint}"
    headers = {'X-API-Key': api_key}

    while True:
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 429:
            reset_timestamp = int(response.headers.get("X-Api-Limit-Reset-Timestamp", 0))
            current_time = int(time.time() * 1000)
            wait_time = (reset_timestamp - current_time) / 1000
            if wait_time > 0:
                print(f"Rate limit exceeded. Waiting for {wait_time:.2f} seconds.")
                time.sleep(wait_time)
            else:
                time.sleep(1)  
        else:
            response.raise_for_status()  
            data_dict = response.json()
            data = data_dict.get('data', [])

            # Convert to DataFrame
            df = pd.DataFrame(data)

            # Convert timestamp to datetime and set as index
            if 'start_time' in df.columns:
                df['timestamp'] = pd.to_datetime(df['start_time'], unit='ms')
                df.set_index('timestamp', inplace=True)

            return df