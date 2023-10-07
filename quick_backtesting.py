import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

asset_1 = pd.read_csv('asset_1.csv', index_col='date')
asset_2 = pd.read_csv('asset_2.csv', index_col='date')
######################
TOTAL_CAPITAL = 10000000
CONTRACT_SIZE = 1000
T_COST = 0.01

index = asset_1.index.difference(asset_2.index)
# Drop this data point, it seems wrong. Further investigation later.
asset_1.drop(['2022-06-20'], inplace=True)

asset_1.rename(columns={"close": "asset_1_close", "nwone_ticker": "asset_1_ticker"}, inplace=True)
asset_2.rename(columns={"close": "asset_2_close", "nwone_ticker": "asset_2_ticker"}, inplace=True)
assets = pd.concat([asset_1["asset_1_close"], asset_1["asset_1_ticker"], asset_2["asset_2_close"], asset_2["asset_2_ticker"]], axis=1)
assets['contract_change'] = assets['asset_2_ticker'] != assets['asset_2_ticker'].shift(1)

train_df = assets.loc[:'2014-01-01',:].copy()
test_df = assets.loc["2014-01-01":, :].copy()
######################
def alpha_signal(df, rolling_window=10):
    df['raw_signal'] = df['asset_2_close']/df['asset_1_close']
    df["alpha_signal"] = df['raw_signal'].rolling(rolling_window).apply(lambda x:(x[-1] - x.mean())/x.std())
    return

def signal_to_portfolio(df):
    df['bet_size'] = 0
    df.loc[(df['alpha_signal'] >= 0.5) &  (df['alpha_signal'] < 1), 'bet_size'] = -0.01 
    df.loc[(df['alpha_signal'] >= 1) & (df['alpha_signal'] < 2), 'bet_size'] = -0.03 
    df.loc[df['alpha_signal'] >= 2, 'bet_size'] = -0.06 
    df.loc[(df['alpha_signal'] <= -0.5) & (df['alpha_signal'] > -1), 'bet_size'] = 0.01 
    df.loc[(df['alpha_signal'] <= -1) & (df['alpha_signal'] > -2), 'bet_size'] = 0.03 
    df.loc[df['alpha_signal'] <= -2, 'bet_size'] = 0.06 
    
    df['position'] = 0
    start_index = df.index[0]
    for i, row in df.iterrows():
        if row['contract_change'] == True or i == df.index[-1]:
            df.loc[start_index: i, 'position'] = int(TOTAL_CAPITAL * df.loc[start_index,'bet_size'] / (df.loc[start_index, "asset_2_close"] * CONTRACT_SIZE))
            start_index = i
    return

def portfolio_to_PnL(df):
    df['dollar_PnL'] = 0
    df["dollar_PnL"] = df['position'].shift(1) * CONTRACT_SIZE * (df['asset_2_close'] - df["asset_2_close"].shift(1))
    df.loc[df['contract_change'] == True, "dollar_PnL"] = 0
    df['dollar_PnL'] -= abs(df['position'] - df['position'].shift(1)) * CONTRACT_SIZE * T_COST
    return

def performance_summary(df): 
    df['total_PnL'] = df['dollar_PnL'].cumsum()
    df['cummax'] = df['total_PnL'].cummax()
    df['drawdown'] = df['total_PnL'] - df['cummax']
    max_drawdown = df['drawdown'].min()
    annualized_dollar_return = df['total_PnL'].iloc[-1] * 252 / len(df)
    annualized_dollar_volatility = df['dollar_PnL'].std() * np.sqrt(252)
    sharpe_ratio = annualized_dollar_return / annualized_dollar_volatility
    return sharpe_ratio, annualized_dollar_return, annualized_dollar_volatility, max_drawdown

def backtesting(df, rollingwindows, folder_name="train_results"):
    PnLs = pd.DataFrame([])
    PnL_summary = pd.DataFrame(columns=['rolling_window', 'sharpe_ratio', 'annualized_dollar_return', 'annualized_dollar_volatility', 'max_dollar_drawdown'])
    for rolling_window in rollingwindows:
        alpha_signal(df, rolling_window)
        signal_to_portfolio(df)
        portfolio_to_PnL(df)
        sharpe_ratio, annualized_dollar_return, annualized_dollar_volatility, max_drawdown = performance_summary(df)
        PnLs = pd.concat([PnLs, df['total_PnL']], axis=1)
        PnLs.rename(columns={'total_PnL': rolling_window}, inplace=True)
        new_row = pd.DataFrame({'rolling_window':[rolling_window], 'sharpe_ratio':[sharpe_ratio], 'annualized_dollar_return':[annualized_dollar_return], 'annualized_dollar_volatility':[annualized_dollar_volatility], 'max_dollar_drawdown':[max_drawdown]})
        PnL_summary = pd.concat([PnL_summary, new_row], ignore_index=True)
    PnL_summary.set_index('rolling_window', inplace=True)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    PnLs.plot(figsize=(20, 12), title=' dollar PnL with Different Rolling Windows', fontsize=14)
    plt.savefig(os.path.join(folder_name, 'strategy_PnL.png'))
    plt.close()
    PnLs.to_csv(os.path.join(folder_name, 'strategy_PnLs.csv'))
    PnL_summary.to_csv(os.path.join(folder_name, 'strategy_summary.csv'))
    return PnLs, PnL_summary
####################

ROLLING_WINDOWS = [5, 10, 20, 30, 100]
backtesting(train_df, ROLLING_WINDOWS)
backtesting(test_df, ROLLING_WINDOWS, folder_name="test_results")
    

