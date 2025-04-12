from ta.volatility import BollingerBands, AverageTrueRange
from hmmlearn import hmm
from ta.trend import ADXIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator, ROCIndicator, StochasticOscillator
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import pickle
import numpy as np

def detect_market_regime(df, n_regimes=5, training_percentage=0.8):
    """
    Detect market regimes using HMM and technical indicators.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with OHLCV and on-chain data
    n_regimes : int
        Number of regimes to detect
    training_percentage : float
        Percentage of data to use for training
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with regime classifications
    """
    print("Starting market regime detection...")
    
    # Validate input data length
    min_required_length = 200  # Based on the largest window size (EMA 200)
    if len(df) < min_required_length:
        raise ValueError(f"Need at least {min_required_length} data points, got {len(df)}")
    
    print("Calculating technical indicators...")
    # Calculate trend indicators using EMA
    df['ema_50'] = EMAIndicator(close=df['close'], window=50).ema_indicator()
    df['ema_100'] = EMAIndicator(close=df['close'], window=100).ema_indicator()
    df['ema_200'] = EMAIndicator(close=df['close'], window=200).ema_indicator()
    
    # MACD (Moving Average Convergence Divergence)
    macd = MACD(close=df['close'], window_slow=26, window_fast=12, window_sign=9)
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()
    
    # Momentum indicators
    df['rsi'] = RSIIndicator(close=df['close'], window=14).rsi()
    df['roc_12'] = ROCIndicator(close=df['close'], window=12).roc()
    
    # Stochastic Oscillator
    stoch = StochasticOscillator(high=df['high'], low=df['low'], close=df['close'], window=14, smooth_window=3)
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()
    
    # Trend strength
    df['adx'] = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14).adx()
    
    # Volatility indicators
    df['atr'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range()
    bb = BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_middle'] = bb.bollinger_mavg()
    df['bb_lower'] = bb.bollinger_lband()
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    
    print("Calculating derived features...")
    # Calculate derived features 
    # Trend direction and strength
    df['ema_50_200_ratio'] = df['ema_50'] / df['ema_200']
    df['ema_50_200_cross'] = np.where(df['ema_50'] > df['ema_200'], 1, -1)
    df['price_to_ema200'] = df['close'] / df['ema_200']
    
    # Momentum-based features
    df['rsi_smoothed'] = df['rsi'].rolling(3).mean()
    df['rsi_trend'] = df['rsi_smoothed'] - df['rsi_smoothed'].shift(5)
    
    # Volatility features
    df['recent_volatility'] = df['close'].pct_change().rolling(5).std() * 100
    df['long_volatility'] = df['close'].pct_change().rolling(30).std() * 100
    df['volatility_ratio'] = df['recent_volatility'] / df['long_volatility']
    
    # Volume analysis
    df['volume_ema'] = EMAIndicator(close=df['volume'], window=20).ema_indicator()
    df['volume_ratio'] = df['volume'] / df['volume_ema']
    
    # Price action
    df['returns'] = df['close'].pct_change().fillna(0)
    df['returns_5h'] = df['close'].pct_change(5).fillna(0)
    df['returns_20h'] = df['close'].pct_change(20).fillna(0)
    
    # Trend persistence
    df['up_days'] = np.where(df['returns'] > 0, 1, 0)
    df['up_streak'] = df['up_days'].groupby((df['up_days'] != df['up_days'].shift(1)).cumsum()).cumsum()
    df['down_days'] = np.where(df['returns'] < 0, 1, 0)
    df['down_streak'] = df['down_days'].groupby((df['down_days'] != df['down_days'].shift(1)).cumsum()).cumsum()
    
    # On-chain specific features 
    if 'inflow_total' in df.columns and 'outflow_total' in df.columns:
        df['net_flow'] = df['inflow_total'] - df['outflow_total']
        df['flow_ratio'] = df['inflow_total'] / df['outflow_total'].replace(0, 1e-8)
    
    # Select features for market regime detection
    feature_columns = [
        # Trend features
        'ema_50_200_ratio', 'ema_50_200_cross', 'price_to_ema200', 'macd', 'macd_diff',
        
        # Momentum features
        'rsi', 'rsi_trend', 'roc_12', 'stoch_k', 'stoch_d',
        
        # Volatility features
        'atr', 'bb_width', 'volatility_ratio',
        
        # Volume and price action
        'volume_ratio', 'returns', 'returns_5h', 'returns_20h',
        'up_streak', 'down_streak',
        
        # Trend strength
        'adx'
    ]
    
    # Add on-chain features 
    if 'inflow_total' in df.columns:
        feature_columns.extend(['net_flow', 'flow_ratio'])
    
    if 'hashrate' in df.columns:
        feature_columns.append('hashrate')
    
    if 'exchange_whale_ratio' in df.columns:
        feature_columns.append('exchange_whale_ratio')
    
    # Create feature dataframe
    features = df[feature_columns].copy()
    
    print("Cleaning data and handling missing values...")
    # Fill missing values
    features = features.ffill().bfill()
    
    # Replace inf/NaN values
    features = features.replace([np.inf, -np.inf], np.nan)
    
    # Calculate means for each column to use for filling NaNs
    feature_means = features.mean()
    features = features.fillna(feature_means)
    
    print("Splitting data for training...")
    # Split data (80% train, 20% test)
    split_idx = int(len(features) * training_percentage)
    train_features = features.iloc[:split_idx]
    test_features = features.iloc[split_idx:]
    
    print("Scaling features...")
    # Scale features
    scaler = StandardScaler().fit(train_features)
    scaled_train = scaler.transform(train_features)
    scaled_test = scaler.transform(test_features)
    scaled_all = scaler.transform(features)
    
    print("Applying PCA...")
    # Apply PCA for dimensionality reduction
    n_components = min(8, len(feature_columns))  
    pca = PCA(n_components=n_components)
    pca_train = pca.fit_transform(scaled_train)
    pca_test = pca.transform(scaled_test)
    pca_all = pca.transform(scaled_all)
    
    print("Training HMM model...")
    # Train the HMM model
    model = hmm.GaussianHMM(
        n_components=n_regimes,
        covariance_type='full',
        n_iter=3000,
        random_state=42,
        tol=1e-5,
        init_params='kmeans'  
    )
    
    model.fit(pca_train)
    
    print("Predicting regimes...")
    # Get regime predictions
    train_regimes = model.predict(pca_train)
    test_regimes = model.predict(pca_test)
    all_regimes = model.predict(pca_all)
    
    # Add regime predictions to the dataframe
    df['regime'] = -1  
    df.iloc[:split_idx, df.columns.get_loc('regime')] = train_regimes
    df.iloc[split_idx:, df.columns.get_loc('regime')] = test_regimes
    
    print("Analyzing regime characteristics...")
    # Calculate regime properties to label them appropriately
    regime_properties = {}
    
    for regime in range(n_regimes):
        mask = df['regime'] == regime
        regime_data = df[mask]
        
        # Skip if no data for this regime
        if len(regime_data) == 0:
            continue
        
        # Calculate key metrics for this regime
        avg_return = regime_data['returns'].mean() * 100
        avg_5d_return = regime_data['returns_5h'].mean() * 100
        avg_volatility = regime_data['recent_volatility'].mean()
        avg_trend = regime_data['ema_50_200_cross'].mean()
        avg_volume = regime_data['volume_ratio'].mean()
        avg_rsi = regime_data['rsi'].mean()
        
        regime_properties[regime] = {
            'return': avg_return,
            'return_5d': avg_5d_return,
            'volatility': avg_volatility,
            'trend': avg_trend,
            'volume': avg_volume,
            'rsi': avg_rsi,
            'count': len(regime_data),
            'pct': len(regime_data) / len(df) * 100
        }
    
    # Sort regimes by return for labeling
    sorted_regimes = sorted(regime_properties.keys(), key=lambda x: regime_properties[x]['return'])
    
    labels = {} 
    
    # Create labels based on regime count
    if n_regimes == 3:
        labels = {
            sorted_regimes[0]: 'bearish',
            sorted_regimes[1]: 'sideways',
            sorted_regimes[2]: 'bullish'
        }
    elif n_regimes == 5:
        labels = {
            sorted_regimes[0]: 'strongly_bearish',
            sorted_regimes[1]: 'mildly_bearish',
            sorted_regimes[2]: 'sideways',
            sorted_regimes[3]: 'mildly_bullish',
            sorted_regimes[4]: 'strongly_bullish'
        }
    else:
        # For other regime counts, create generic labels
        labels = {}
        for i, regime in enumerate(sorted_regimes):
            if i == 0:
                labels[regime] = 'strongly_bearish'
            elif i == len(sorted_regimes) - 1:
                labels[regime] = 'strongly_bullish'
            elif i < len(sorted_regimes) // 2:
                labels[regime] = f'bearish_{i}'
            elif i > len(sorted_regimes) // 2:
                labels[regime] = f'bullish_{i - len(sorted_regimes) // 2}'
            else:
                labels[regime] = 'sideways'
    
    # Add regime labels to dataframe
    df['regime_label'] = df['regime'].map(labels)

    
    # Save the model 
    print("Saving model...")
    model_data = {
        'model': model,
        'pca': pca,
        'scaler': scaler,
        'n_components': n_regimes,
        'feature_columns': feature_columns,
        'labels': labels
    }
    
    with open("hmm_market_regime_model.pkl", 'wb') as file:
        pickle.dump(model_data, file)
    
    print("Market regime detection completed.")
    return df, model_data
