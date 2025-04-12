import plotly.graph_objects as go
import numpy as np

def backtest(df, model_data, initial_cash=10000):
    """
    Backtest a trading strategy based on signals in the dataframe.

    """
    
    # Generate results
    results_df, transactions = generate_and_execute_trades(
        df=df,
        model_data=model_data
    )
    
    # Plot results
    plot_backtest_results(results_df)
    
    # Calculate and print performance metrics
    print_performance_metrics(results_df, initial_cash, transactions)
    
    return df


def generate_and_execute_trades(df, model_data, initial_cash=10000):
    """
    Generates trading signals and executes trades with portfolio tracking and profit counting
    Returns:
        results_df: DataFrame with all signals and portfolio values
        transactions: List of all executed trades
        performance_stats: Dictionary of performance metrics
    """
    print("Generating and executing trades...")
    
    # Initialize trading parameters
    FEE_PCT = 0.06
    RISK_PER_TRADE = 0.5  # 50% of cash per trade
    labels = model_data['labels']
    
    # Initialize tracking variables
    df = df.copy()
    df['signal'] = 0
    current_position = 0  # -1=short, 0=flat, 1=long
    position_entry_price = None
    
    # Portfolio tracking
    cash = initial_cash
    position = 0  # Number of units held (positive for long, negative for short)
    portfolio_values = []
    transactions = []
    
    # Performance tracking
    stats = {
        'total_trades': 0,
        'long_trades': 0,
        'short_trades': 0,
        'profitable_long': 0,
        'unprofitable_long': 0,
        'profitable_short': 0,
        'unprofitable_short': 0,
        'max_portfolio_value': initial_cash,
        'min_portfolio_value': initial_cash
    }
    
    bullish_regimes = [regime for regime, label in labels.items() if 'bullish' in label]
    bearish_regimes = [regime for regime, label in labels.items() if 'bearish' in label]
    sideways_regimes = [regime for regime, label in labels.items() if 'sideways' in label]

    current_position = 0

    for i in range(1, len(df)):
        
        #################### Signal Generation ####################  
        fee_buffer = df['close'].iloc[i] * FEE_PCT * 2
        prev_regime = df['regime'].iloc[i-1]
        curr_regime = df['regime'].iloc[i]
        
        # Reset signal for current bar
        df.loc[df.index[i], 'signal'] = 0
        
        # Entry conditions
        at_upper = ((df['high'].iloc[i] > df['bb_upper'].iloc[i]) & 
                   (df['rsi'].iloc[i] > 70) & 
                   ((df['bb_upper'].iloc[i] - df['bb_middle'].iloc[i]) > fee_buffer))
        at_lower = ((df['low'].iloc[i] < df['bb_lower'].iloc[i]) & 
                   (df['rsi'].iloc[i] < 30) & 
                   ((df['bb_middle'].iloc[i] - df['bb_lower'].iloc[i]) > fee_buffer))
        
        # Generate signals
        if current_position == 0:  # Flat position
            # Sideways BB strategy
            if (curr_regime in sideways_regimes):
                if at_upper:
                    df.loc[df.index[i], 'signal'] = -1
                elif at_lower:
                    df.loc[df.index[i], 'signal'] = 1
            
            # Trend strategy
            elif (prev_regime != curr_regime):
                if (curr_regime in bullish_regimes):
                    df.loc[df.index[i], 'signal'] = 1
                elif (curr_regime in bearish_regimes):
                    df.loc[df.index[i], 'signal'] = -1
        
        # Exit conditions (always allowed)
        elif current_position > 0:  # Long position
            exit_condition = (
                (df['close'].iloc[i] >= (position_entry_price * (1 + FEE_PCT))) or
                (curr_regime in bearish_regimes)
            )
            if exit_condition:
                df.loc[df.index[i], 'signal'] = -1
        
        elif current_position < 0:  # Short position
            exit_condition = (
                (df['close'].iloc[i] <= (position_entry_price * (1 + FEE_PCT))) or
                (curr_regime in bullish_regimes)
            )
            if exit_condition:
                df.loc[df.index[i], 'signal'] = 1

        

        #################### Trade Execution ####################  
        signal = df.loc[df.index[i], 'signal']
        price = df['close'].iloc[i]
        
        if signal == 1 and cash > 0 and current_position == 0:  # Buy entry (Long)
            # Buy execution
            buy_amount = cash * RISK_PER_TRADE
            trading_fee = buy_amount * FEE_PCT
            num_shares = buy_amount / price
            position += num_shares
            cash -= (num_shares * price) + trading_fee
            current_position = 1
            position_entry_price = price
            stats['long_trades'] += 1
            stats['total_trades'] += 1
            print(f"{df.index[i]}: BUY {num_shares:.4f} units at {price:.2f}")
        
        elif signal == -1 and cash > 0 and current_position == 1:  # Sell exit (Long exit)
            # Sell execution
            sell_value = position * price
            num_shares = position
            trading_fee = sell_value * FEE_PCT
            cash += (sell_value - trading_fee)
            position = 0
            current_position = 0
            
            # Calculate Profit & Loss
            pnl = ((price - position_entry_price) / position_entry_price - FEE_PCT) * 100
            if pnl > 0:
                stats['profitable_long'] += 1
                pnl_str = f"+{pnl:.2f}%"
            else:
                stats['unprofitable_long'] += 1
                pnl_str = f"{pnl:.2f}%"
            
            print(f"{df.index[i]}: SELL {num_shares:.4f} units at {price:.2f} ({pnl_str})")
            
        
        
        elif signal == -1 and cash > 0 and current_position == 0:  # Short entry
            # Sell execution
            risk_amount = cash * RISK_PER_TRADE
            num_shares = risk_amount / price
            trading_fee = num_shares * price * FEE_PCT
            cash += (num_shares * price) - trading_fee 
            position = -num_shares
            current_position = -1
            position_entry_price = price
            stats['short_trades'] += 1
            stats['total_trades'] += 1
            print(f"{df.index[i]}: SHORT {num_shares:.4f} units at {price:.2f}")
        
        elif signal == 1 and cash > 0 and current_position == -1:  # Short cover (Short exit)
            # Buy back execution
            num_shares = abs(position)
            trading_fee = price * num_shares * FEE_PCT
            cash -= (num_shares * price) + trading_fee
            current_position = 0
            position_entry_price = price
            
            # Calculate Profit & Loss
            pnl = ((position_entry_price - price) / position_entry_price - FEE_PCT) * 100
            
            if pnl > 0:
                stats['profitable_short'] += 1
                pnl_str = f"+{pnl:.2f}%"
            else:
                stats['unprofitable_short'] += 1
                pnl_str = f"{pnl:.2f}%"
            
            print(f"{df.index[i]}: COVER {num_shares:.4f} units at {price:.2f} ({pnl_str})")
            
            # Reset tracking variables
            position = 0
            current_position = 0
            
        
        #################### Portfolio Tracking ####################  
        if position > 0:  # Long position
            position_value = position * price
            current_value = cash + position_value
        elif position < 0:  # Short position
            position_liability = abs(position) * price
            current_value = cash - position_liability
        else:  # No position
            current_value = cash
        
        portfolio_values.append(current_value)
        
        # Update performance stats
        stats['max_portfolio_value'] = max(stats['max_portfolio_value'], current_value)
        stats['min_portfolio_value'] = min(stats['min_portfolio_value'], current_value)
    
    # Add portfolio values to dataframe
    df['portfolio_value'] = [initial_cash] + portfolio_values  
    df['portfolio_return'] = df['portfolio_value'].pct_change().fillna(0)
    
    # Calculate final performance metrics
    final_value = df['portfolio_value'].iloc[-1]
    stats['total_return'] = (final_value - initial_cash) / initial_cash * 100
    stats['long_win_rate'] = stats['profitable_long'] / max(1, stats['long_trades']) * 100
    stats['short_win_rate'] = stats['profitable_short'] / max(1, stats['short_trades']) * 100
    stats['overall_win_rate'] = (stats['profitable_long'] + stats['profitable_short']) / max(1, stats['total_trades']) * 100
    stats['max_drawdown'] = (stats['max_portfolio_value'] - stats['min_portfolio_value']) / stats['max_portfolio_value'] * 100
    
    print("\n=== Performance Summary ===")
    print(f"Initial Capital: ${initial_cash:,.2f}")
    print(f"Final Portfolio Value: ${final_value:,.2f} ({stats['total_return']:.2f}%)")
    print(f"\nLong Trades: {stats['long_trades']}")
    print(f"  Profitable: {stats['profitable_long']} ({stats['long_win_rate']:.1f}%)")
    print(f"  Unprofitable: {stats['unprofitable_long']}")
    print(f"\nShort Trades: {stats['short_trades']}")
    print(f"  Profitable: {stats['profitable_short']} ({stats['short_win_rate']:.1f}%)")
    print(f"  Unprofitable: {stats['unprofitable_short']}")
    print(f"\nOverall Win Rate: {stats['overall_win_rate']:.1f}%")
    print(f"Max Drawdown: {stats['max_drawdown']:.2f}%")
    
    return df, transactions


def plot_backtest_results(df):
    
    print("Generating graph...")
    fig = go.Figure()
    
    # Define regime colors
    regime_colors = {
        'strongly_bearish': '#FF3333',    
        'mildly_bearish': '#FF9933',    
        'sideways': '#B3B3B3',         
        'mildly_bullish': '#33CC33',     
        'strongly_bullish': '#006400'      
    }

    # Plot price data
    fig.add_trace(go.Scatter(
        x=df.index, y=df['close'], 
        mode='lines', name='Price',
        line=dict(color='rgba(0,0,255,0.3)', width=1)
    ))
    
    # Plot regime segments 
    if 'regime_label' in df.columns:
        df['regime_change'] = df['regime_label'] != df['regime_label'].shift(1)
        df['group'] = df['regime_change'].cumsum()
        
        seen_regimes = set()
        # Plot each segment
        for _, group_data in df.groupby('group'):
            regime = group_data['regime_label'].iloc[0]
            show_legend = regime not in seen_regimes
            seen_regimes.add(regime)
            fig.add_trace(go.Scatter(
                x=group_data.index,
                y=group_data['close'],
                mode='lines',
                line=dict(color=regime_colors.get(regime, 'gray')),
                name=regime.replace('_', ' ').title(),
                legendgroup=regime,
                showlegend=show_legend
            ))
   
    # Add portfolio value on secondary axis
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['portfolio_value'],
        yaxis='y2',
        name='Portfolio',
        line=dict(color='purple', width=2)
    ))
    
    # Add buy/sell markers
    buy_added = False
    sell_added = False
    
    for i in range(len(df)):
        signal = df['signal'].iloc[i]
        price = df['close'].iloc[i]
        
        if signal == 1:  # Buy signal
            showlegend = not buy_added
            buy_added = True
            
            fig.add_trace(go.Scatter(
                x=[df.index[i]], 
                y=[price], 
                mode='markers', 
                name='Buy Signal',
                marker=dict(symbol='triangle-up', color='green', size=10),
                showlegend=showlegend,
                legendgroup='buy'
            ))
        
        elif signal == -1:  # Sell signal
            showlegend = not sell_added
            sell_added = True
            
            fig.add_trace(go.Scatter(
                x=[df.index[i]], 
                y=[price], 
                mode='markers', 
                name='Sell Signal',
                marker=dict(symbol='triangle-down', color='red', size=10),
                showlegend=showlegend,
                legendgroup='sell'
            ))
    
    # Update layout
    fig.update_layout(
        title="Trading Strategy Backtest",
        xaxis_title="Date",
        yaxis=dict(
            title="Price",
            side="left"
        ),
        yaxis2=dict(
            title="Portfolio Value",
            overlaying='y',
            side='right',
            showgrid=False
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Show the plot
    fig.show()


def calculate_mdd(values):
    # Convert to numpy array if it's not already
    values = np.array(values)
    
    # Calculate the running maximum
    running_max = np.maximum.accumulate(values)
    
    # Calculate drawdown
    drawdown = (values - running_max) / running_max
    
    # Find the maximum drawdown
    mdd = np.min(drawdown)
    
    return mdd


def print_performance_metrics(df, initial_cash, transactions):
    
    # Calculate basic returns
    final_portfolio = df['portfolio_value'].iloc[-1]
    total_return = (final_portfolio / initial_cash - 1) * 100
    
    # Calculate daily PnL and returns
    df['daily_pnl'] = df['portfolio_value'].diff().fillna(0)
    df['returns'] = df['portfolio_value'].pct_change().fillna(0)
    
    # Sharpe Ratio 
    sharpe_ratio = np.sqrt(252) * df['returns'].mean() / df['returns'].std() if df['returns'].std() > 0 else 0
    
    # Max Drawdown
    df['peak'] = df['portfolio_value'].cummax()
    df['drawdown'] = (df['peak'] - df['portfolio_value']) / df['peak']
    max_drawdown = df['drawdown'].max() * 100  # as percentage
    
    # Win rate calculation
    long_trades = 0
    short_trades = 0
    profitable_long = 0
    profitable_short = 0
    
    for i in range(1, len(transactions)):
        prev_t = transactions[i-1]
        curr_t = transactions[i]
        
        # Long trades (BUY -> SELL)
        if "BUY" in prev_t and "SELL" in curr_t:
            long_trades += 1
            buy_price = float(prev_t.split("at")[1].split()[0])
            sell_price = float(curr_t.split("at")[1].split()[0])
            if sell_price > buy_price:
                profitable_long += 1
                
        # Short trades (SHORT -> COVER)
        elif "SHORT" in prev_t and "COVER" in curr_t:
            short_trades += 1
            short_price = float(prev_t.split("at")[1].split()[0])
            cover_price = float(curr_t.split("at")[1].split()[0])
            if cover_price < short_price:
                profitable_short += 1
    
    # Trade metrics
    total_trades = long_trades + short_trades
    win_rate = (profitable_long + profitable_short) / total_trades * 100 if total_trades > 0 else 0
    
    # Compile all metrics
    metrics = {
        'total_return (%)': total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown (%)': max_drawdown,
        'win_rate (%)': win_rate,
        'num_trades': total_trades,
        'profitable_long': profitable_long,
        'profitable_short': profitable_short,
        'long_trades': long_trades,
        'short_trades': short_trades
    }
    
    # Print formatted results
    print("\n===== PERFORMANCE METRICS =====")
    print(f"{'Total Return:':<20} {metrics['total_return (%)']:.2f}%")
    print(f"{'Sharpe Ratio:':<20} {metrics['sharpe_ratio']:.2f}")
    print(f"{'Max Drawdown:':<20} {metrics['max_drawdown (%)']:.2f}%")
    print(f"{'Win Rate:':<20} {metrics['win_rate (%)']:.1f}%")
    print(f"{'Total Trades:':<20} {metrics['num_trades']}")
    print(f"\nLong Trades: {metrics['long_trades']} (Profitable: {metrics['profitable_long']})")
    print(f"Short Trades: {metrics['short_trades']} (Profitable: {metrics['profitable_short']})")
    
    return metrics
