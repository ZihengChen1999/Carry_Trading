import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

asset_1 = pd.read_csv('asset_1.csv', index_col='date')
asset_2 = pd.read_csv('asset_2.csv', index_col='date')

index = asset_1.index.difference(asset_2.index)
# Drop this data point, it seems wrong. Further investigation later.
asset_1.drop(['2022-06-20'], inplace=True)

asset_1.rename(columns={'nwone_ticker':'ticker_1', 'close':'close_1'}, inplace=True)
asset_2.rename(columns={'nwone_ticker':'ticker_2', 'close':'close_2'}, inplace=True)

assets = pd.concat([asset_1['ticker_1'], asset_2['ticker_2'], asset_1['close_1'], asset_2['close_2']], axis=1)

assets['contract_change'] = assets['ticker_2'] != assets['ticker_2'].shift(1)

TOTAL_CAPITAL = 10000000
CONTRACT_SIZE = 1000
T_Cost = 0.01
#####################

train_df = assets.loc[:'2014-01-01', :].copy()
test_df = assets.loc['2014-01-01':, :].copy()

train_df
test_df

def signal(df, rolling_window=20):
    df['raw_signal'] = df['close_2'] / df['close_1']
    df['alpha_signal'] = df['raw_signal'].rolling(rolling_window).apply(lambda x:(x[-1] - x.mean())/x.std())
    return

def signal_to_portfolio(df):
    df['bet_size'] = 0
    df.loc[(df['alpha_signal'] >= 0.5) & (df['alpha_signal'] < 1), 'bet_size'] = -0.01
    df.loc[(df['alpha_signal'] >= 1) & (df['alpha_signal'] < 2), 'bet_size'] = -0.03
    df.loc[df['alpha_signal'] >= 2, 'bet_size'] = -0.06
    df.loc[(df['alpha_signal'] <= -0.5) & (df['alpha_signal'] > -1), 'bet_size'] = 0.01
    df.loc[(df['alpha_signal'] <= -1) & (df['alpha_signal'] > -2), 'bet_size'] = 0.03
    df.loc[df['alpha_signal'] <= -2, 'bet_size'] = 0.06


    df['position'] = 0
    start_index = df.index[0]
    for i, row in df.iterrows():
        if row['contract_change'] == True or i == df.index[-1]:
            df.loc[start_index:i, 'position'] = int(TOTAL_CAPITAL * df.loc[start_index, 'bet_size']/(df.loc[start_index, 'close_2'] * CONTRACT_SIZE))
            start_index = i
    return 

def portfolio_to_PnL(df):
    df['PnL'] = 0
    df['PnL'] = df['position'].shift(1) * (df['close_2'] - df['close_2'].shift(1)) * CONTRACT_SIZE
    df.loc[df['contract_change'] == True, 'PnL'] = 0
    df['PnL'] -= abs(df['position'] - df['position'].shift(1)) * CONTRACT_SIZE * T_Cost
    return

def performance_summary(df):
    df['total_PnL'] = df['PnL'].cumsum()
    df['cummax'] = df['total_PnL'].cummax()
    df['drawdown'] = df['total_PnL'] - df['cummax']
    max_drawdown = df['drawdown'].min()
    ann_dollar_return = df['total_PnL'].iloc[-1] * 252 / len(df)
    ann_dollar_std = df['PnL'].std() * 16
    sharpe_ratio = ann_dollar_return / ann_dollar_std
    return sharpe_ratio, max_drawdown, ann_dollar_return, ann_dollar_std 

def backtest(df, rolling_windows, folder_name):
    PnLs = pd.DataFrame([])
    PnL_Summary = pd.DataFrame([])
    for rolling_window in rolling_windows:
        signal(df, rolling_window)
        signal_to_portfolio(df)
        print(df.tail(30))
        portfolio_to_PnL(df)
        sharpe_ratio, max_drawdown, ann_dollar_return, ann_dollar_std = performance_summary(df)
        new_row = pd.DataFrame({'rolling_window': [rolling_window], 'sharpe_ratio': [sharpe_ratio], 'ann_dollar_return': [ann_dollar_return], 'ann_dollar_std': [ann_dollar_std], 'max_drawdown': [max_drawdown]})
        PnL_Summary = pd.concat([PnL_Summary, new_row], ignore_index=True)
        PnLs = pd.concat([PnLs, df['total_PnL']], axis=1)
        PnLs.rename(columns={'total_PnL': rolling_window}, inplace=True)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    PnLs.plot(figsize=(20, 12), title= 'PnLs')
    plt.savefig(os.path.join(folder_name, 'PnLs.jpg'))
    PnLs.to_csv(os.path.join(folder_name, 'PnLs.csv'))
    PnL_Summary.to_csv(os.path.join(folder_name, 'PnL_Summary.csv'))
    return

ROLLING_WINDOWS = [5, 10, 20, 50, 100]

backtest(train_df, ROLLING_WINDOWS, 'practice_train')
backtest(test_df, ROLLING_WINDOWS, 'practice_test')