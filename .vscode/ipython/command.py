import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
# One futures, multiple maturity dates, time-series trading strategy, according to our theory, the sharper the term structure is, the heavier we woud like to short it/ we predict a negative return on asset 2
# Maximizing for Sharpe ratio
# Rebalance when we change contract, aounrd 4-weeks, we close the position using the last close price before changing and open the position using the first close price after changing, there is one day gap between the two positions
# Trading signal/Alpha: Using the rolling Z-score of asset_2/assets_1 as trading signal to trade asset_2
# Bet sizing: Assuming we have a total of $1,000,000 to invest
###############################################################################################################################################################
# read in the data
asset_1 = pd.read_csv('asset_1.csv', index_col='date')
asset_2 = pd.read_csv('asset_2.csv', index_col='date')
if not os.path.exists('plots'):
    os.makedirs('plots')
asset_1.describe()
asset_2.describe()
# contract size, tcost and value1pt are all constants, so we do not need to keep them iun the dataframe
# Rename asset_1 close column to asset_1_close and asset_2 close column to asset_2_close
asset_1.rename(columns={'close': 'asset_1_close'}, inplace=True)
asset_2.rename(columns={'close': 'asset_2_close'}, inplace=True)
asset_1.rename(columns={'nwone_ticker': 'asset_1_nwone_ticker'}, inplace=True)
asset_2.rename(columns={'nwone_ticker': 'asset_2_nwone_ticker'}, inplace=True)
# Finding out the different index between asset_1 and asset_2
asset_1_index = asset_1.index
asset_2_index = asset_2.index
asset_1_index.difference(asset_2_index)
# After checking the data, the data point of 2022-06-20 within asset_1 is highly suspectable, it is using U2022, which is the same as the contract in our asset_2, so I will drop this data point for now, and we can do further investigation later
# Drop the data point of 2022-06-20 within asset_1
asset_1.drop('2022-06-20', inplace=True)
# Combine close price of asset_1 and asset_2 into one dataframe
asset = pd.concat([asset_1['asset_1_close'], asset_2['asset_2_close'], asset_1['asset_1_nwone_ticker'], asset_2['asset_2_nwone_ticker']], axis=1)
# Seperate the dataframe into training and testing set, before and after 2014-01-01
asset_train = asset.loc[:'2014-01-01'].copy()
asset_test = asset.loc['2014-01-01':].copy()
###############################################################################################################################################################
# trading signal building
def data_prep(asset_train):
    asset_train['asset_2/asset_1'] = asset_train['asset_2_close'] / asset_train['asset_1_close']
    asset_train['asset_2_return'] = asset_train['asset_2_close'].pct_change()
    # Flag the time when we change contract
    asset_train['contract_change'] = asset_train['asset_2_nwone_ticker'] != asset_train['asset_2_nwone_ticker'].shift(1)
    asset_train.loc[asset_train['contract_change'] == True, 'asset_2_return'] = 0
    # Add a column showing the final price until we the last day we use the same contract as asset_2
    asset_train['asset_2_price_before_changing'] = 0
    price = 0
    start_date_index = asset_train.index[0]
    for i, row in asset_train.iterrows():
        if row['contract_change'] == 1 or i == asset_train.index[-1]:
            asset_train.loc[start_date_index:i, 'asset_2_price_before_changing'] = price
            start_date_index = i
        price = row['asset_2_close']
    asset_train['asset_2_return_before_changing'] = asset_train['asset_2_price_before_changing'] / asset_train['asset_2_close'] - 1
    return  asset_train
def signal_IC(asset_train, rolling_window = 252):
    # We calculate three types of predicting power for our alpha signal:
    #     1) Daily predicting power: the correlation between the bext daily return and today's Z-score
    #     2) Contract changing predicting power: the correlation between the return until when we change contract as asset1 and asset 2 and today's Z-score, using every day
    #     3) Rebalancing day predicting power: At the date we change contract and open new position, the correlation between Z-score and the return until when we change contract
    asset_train['z_score'] = asset_train['asset_2/asset_1'].rolling(rolling_window).apply(lambda x: (x[-1] - x.mean()) / x.std())
    daily_IC = asset_train['z_score'].corr(asset_train['asset_2_return'].shift(-1))
    before_changing_IC = asset_train['z_score'].corr(asset_train['asset_2_return_before_changing'])
    changing_contract_dates = asset_train.loc[asset_train['contract_change'] == 1].copy()
    rebalancing_date_IC = changing_contract_dates['z_score'].corr(changing_contract_dates['asset_2_return_before_changing'])
    return daily_IC, before_changing_IC, rebalancing_date_IC, asset_train
###############################################################################################################################################################
# Sizing and PnL
# If Z-score is [0.5, 1], I will risk 1% of my free capital to short asset_2
# If Z-score is [1, 2], I will risk 3% of my free capital to short asset_2
# If Z-score is [2, x], I will risk 6% of my free capital to short asset_2
# When the Z score is smaller than 0, I will go long, but the bet size is the same as the above
def sizing_and_PnL(prepared_train_with_alpha_signal, TOTAL_CAPITAL, CONTRACT_SIZE):
    # Add a column showing the bet size based on the Z-score
    prepared_train_with_alpha_signal['bet_size'] = 0
    prepared_train_with_alpha_signal.loc[(prepared_train_with_alpha_signal['z_score'] < 1) & (prepared_train_with_alpha_signal['z_score'] > 0.5), 'bet_size'] = -0.01
    prepared_train_with_alpha_signal.loc[(prepared_train_with_alpha_signal['z_score'] < 2) & (prepared_train_with_alpha_signal['z_score'] >= 1.0), 'bet_size'] = -0.03
    prepared_train_with_alpha_signal.loc[prepared_train_with_alpha_signal['z_score'] >= 2, 'bet_size'] = -0.06
    prepared_train_with_alpha_signal.loc[(prepared_train_with_alpha_signal['z_score'] < -0.5) & (prepared_train_with_alpha_signal['z_score'] >= -1), 'bet_size'] = 0.01
    prepared_train_with_alpha_signal.loc[(prepared_train_with_alpha_signal['z_score'] < -1.0) & (prepared_train_with_alpha_signal['z_score'] >= -2), 'bet_size'] = 0.03
    prepared_train_with_alpha_signal.loc[prepared_train_with_alpha_signal['z_score'] < -2, 'bet_size'] = 0.06
    prepared_train_with_alpha_signal['contract_number'] = 0
    # Calculate contracts number as we rebalance on the date we change contracts
    start_date_index = prepared_train_with_alpha_signal.index[0]
    start_row = prepared_train_with_alpha_signal.iloc[0]
    total_contracts_num = 0
    for i, row in prepared_train_with_alpha_signal.iterrows():
        if row['contract_change'] == 1 or i == prepared_train_with_alpha_signal.index[-1]:
            if start_row['bet_size']:
                prepared_train_with_alpha_signal.loc[start_date_index:i, 'contract_number'] = int(TOTAL_CAPITAL * start_row['bet_size'] / (start_row['asset_2_close'] * CONTRACT_SIZE))
            start_date_index = i
            start_row = row
    # Calculate dollar PnL based on the contract number
    prepared_train_with_alpha_signal['dollar_PnL'] = prepared_train_with_alpha_signal['contract_number'].shift(1) * CONTRACT_SIZE * (prepared_train_with_alpha_signal['asset_2_close'] - prepared_train_with_alpha_signal['asset_2_close'].shift(1))
    # When we change contract, pay trading fee for the round trade
    prepared_train_with_alpha_signal.loc[prepared_train_with_alpha_signal['contract_change'] == 1, 'dollar_PnL'] -= TRADING_COST * CONTRACT_SIZE * 2 * abs(prepared_train_with_alpha_signal.loc[prepared_train_with_alpha_signal['contract_change'] == 1, 'contract_number'])
    # Set PnL to be 0 when we change contract
    prepared_train_with_alpha_signal.loc[prepared_train_with_alpha_signal['contract_change'] == 1, 'dollar_PnL'] = 0
    # Calculate the total PnL
    prepared_train_with_alpha_signal['total_PnL'] = prepared_train_with_alpha_signal['dollar_PnL'].cumsum()
    # Caculate the maximum drawdown
    prepared_train_with_alpha_signal['cummax'] = prepared_train_with_alpha_signal['total_PnL'].cummax()
    prepared_train_with_alpha_signal['drawdown'] = prepared_train_with_alpha_signal['total_PnL'] - prepared_train_with_alpha_signal['cummax']
    max_drawdown = prepared_train_with_alpha_signal['drawdown'].min()
    annualized_return = prepared_train_with_alpha_signal['total_PnL'].iloc[-1] * 252 / len(prepared_train_with_alpha_signal)
    annualized_volatility = prepared_train_with_alpha_signal['dollar_PnL'].std() * np.sqrt(252)
    sharpe_ratio = annualized_return / annualized_volatility
    return prepared_train_with_alpha_signal, annualized_return, annualized_volatility, sharpe_ratio, max_drawdown
###############################################################################################################################################################
# In-sample performance analysis: From 2000 to 2014
ROLLING_WINDOWS = [5, 10, 20, 50, 100, 250, 500]
TOTAL_CAPITAL = 10000000
CONTRACT_SIZE = 1000
TRADING_COST = 0.01
# 10 per contract
prepared_train = data_prep(asset_train.copy())
train_strategy_performance = pd.DataFrame(columns=['rolling_window', 'daily_IC', 'before_changing_contract_IC', 'rebalancing_date_IC', 'annualized_dollar_return', 'annualized_dollar_volatility', 'sharpe_ratio', 'max_dollar_drawdown'])
train_strategy_PnL = pd.DataFrame([])
for rolling_window in ROLLING_WINDOWS:
    daily_IC, before_changing_IC, rebalancing_date_IC, prepared_train_with_alpha_signal = signal_IC(prepared_train.copy(), rolling_window)
    train_contracts_and_PnL, annualized_return, annualized_volatility, sharpe_ratio, max_drawdown = sizing_and_PnL(prepared_train_with_alpha_signal.copy(), TOTAL_CAPITAL, CONTRACT_SIZE)
    new_row = pd.DataFrame({'rolling_window': [rolling_window], 'daily_IC': [daily_IC], 'before_changing_contract_IC': [before_changing_IC], 'rebalancing_date_IC': [rebalancing_date_IC], 'annualized_dollar_return': [annualized_return], 'annualized_dollar_volatility': [annualized_volatility], 'sharpe_ratio': [sharpe_ratio], 'max_dollar_drawdown': [max_drawdown]})
    train_strategy_performance = pd.concat([train_strategy_performance, new_row], ignore_index=True)
    # Rename the column of total_PnL to the rolling window
    train_strategy_PnL = pd.concat([train_strategy_PnL, train_contracts_and_PnL['total_PnL']], axis=1)
    train_strategy_PnL.rename(columns={'total_PnL': rolling_window}, inplace=True)
train_contracts_and_PnL.to_csv('train_contracts_and_PnL.csv')
train_strategy_performance.to_csv('train_strategy_performance.csv')
# Plot train strategy PnL, annotating sharpe
train_strategy_PnL.plot(figsize=(20, 12), title='In-sample Strategy dollar PnL with Different Rolling Windows', fontsize=14)
plt.savefig('plots/train_strategy_PnL.png')
plt.close()
###############################################################################################################################################################
# Out-of-sample performance analysis: From 2014 to 2023
ROLLING_WINDOWS = [5, 10, 20, 50, 100, 250, 500]
TOTAL_CAPITAL = 10000000
CONTRACT_SIZE = 1000
TRADING_COST = 0.01
prepared_test = data_prep(asset_test.copy())
test_strategy_performance = pd.DataFrame(columns=['rolling_window', 'daily_IC', 'before_changing_contract_IC', 'rebalancing_date_IC', 'annualized_dollar_return', 'annualized_dollar_volatility', 'sharpe_ratio', 'max_dollar_drawdown'])
test_strategy_PnL = pd.DataFrame([])
for rolling_window in ROLLING_WINDOWS:
    daily_IC, before_changing_IC, rebalancing_date_IC, prepared_test_with_alpha_signal = signal_IC(prepared_test.copy(), rolling_window)
    test_contracts_and_PnL, annualized_return, annualized_volatility, sharpe_ratio, max_drawdown = sizing_and_PnL(prepared_test_with_alpha_signal.copy(), TOTAL_CAPITAL, CONTRACT_SIZE)
    new_row = pd.DataFrame({'rolling_window': [rolling_window], 'daily_IC': [daily_IC], 'before_changing_contract_IC': [before_changing_IC], 'rebalancing_date_IC': [rebalancing_date_IC], 'annualized_dollar_return': [annualized_return], 'annualized_dollar_volatility': [annualized_volatility], 'sharpe_ratio': [sharpe_ratio], 'max_dollar_drawdown': [max_drawdown]})
    test_strategy_performance = pd.concat([test_strategy_performance, new_row], ignore_index=True)
    test_strategy_PnL = pd.concat([test_strategy_PnL, test_contracts_and_PnL['total_PnL']], axis=1)
    test_strategy_PnL.rename(columns={'total_PnL': rolling_window}, inplace=True)
test_contracts_and_PnL.to_csv('test_contracts_and_PnL.csv')
test_strategy_performance.to_csv('test_strategy_performance.csv')
test_strategy_PnL.plot(figsize=(20, 12), title='Out-of-sample Strategy dollar PnL with Different Rolling Windows', fontsize=14)
plt.savefig('plots/test_strategy_PnL.png')

