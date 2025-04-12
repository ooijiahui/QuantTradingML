
from data_processing import fetch_all_data
from features_engineering import detect_market_regime
from backtest import backtest

def main():
    # Step 1: Fetch data
    df = fetch_all_data()
    
    # Step 2: Feature engineering
    df, model_data = detect_market_regime(df)

    # # Step 3: Backtest
    df = backtest(df, model_data)
   
   
if __name__ == '__main__':
    main()

