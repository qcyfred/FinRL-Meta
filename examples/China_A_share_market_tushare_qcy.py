import warnings

warnings.filterwarnings("ignore")

import pandas as pd
from IPython import display

display.set_matplotlib_formats("svg")

from meta import config
from meta.data_processor import DataProcessor
from main import check_and_make_directories
from meta.data_processors.tushare import Tushare, ReturnPlotter
from meta.env_stock_trading.env_stocktrading_China_A_shares import (
    StockTradingEnv,
)
from agents.stablebaselines3_models import DRLAgent
import os
from typing import List
from argparse import ArgumentParser
from meta import config
from meta.config_tickers import DOW_30_TICKER
from meta.config import (
    DATA_SAVE_DIR,
    TRAINED_MODEL_DIR,
    TENSORBOARD_LOG_DIR,
    RESULTS_DIR,
    INDICATORS,
    TRAIN_START_DATE,
    TRAIN_END_DATE,
    TEST_START_DATE,
    TEST_END_DATE,
    TRADE_START_DATE,
    TRADE_END_DATE,
    ERL_PARAMS,
    RLlib_PARAMS,
    SAC_PARAMS,
    ALPACA_API_KEY,
    ALPACA_API_SECRET,
    ALPACA_API_BASE_URL,
)
import pyfolio
from pyfolio import timeseries

pd.options.display.max_columns = None

print("ALL Modules have been imported!")


### Create folders

import os

"""
use check_and_make_directories() to replace the following

if not os.path.exists("./datasets"):
    os.makedirs("./datasets")
if not os.path.exists("./trained_models"):
    os.makedirs("./trained_models")
if not os.path.exists("./tensorboard_log"):
    os.makedirs("./tensorboard_log")
if not os.path.exists("./results"):
    os.makedirs("./results")
"""

check_and_make_directories(
    [DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR]
)


### Download data, cleaning and feature engineering

ticker_list = [
    "600000.SH",
    # "600009.SH",
    # "600016.SH",
    # "600028.SH",
    # "600030.SH",
    # "600031.SH",
    # "600036.SH",
    # "600050.SH",
    # "600104.SH",
    # "600196.SH",
    # "600276.SH",
    # "600309.SH",
    # "600519.SH",
    # "600547.SH",
    # "600570.SH",

    # "000905.SH",
]
# ticker_list = ['600000.XSHG', '600009.XSHG', '600016.XSHG', '600028.XSHG', '600030.XSHG', '600031.XSHG', '600036.XSHG', '600050.XSHG', '600104.XSHG', '600196.XSHG', '600276.XSHG', '600309.XSHG', '600519.XSHG', '600547.XSHG', '600570.XSHG']


TRAIN_START_DATE = "2015-01-01"
TRAIN_END_DATE = "2019-08-01"
TRADE_START_DATE = "2019-08-01"
TRADE_END_DATE = "2020-01-03"


TIME_INTERVAL = "1d"
kwargs = {}
kwargs["token"] = "d820068d786c287ae44926d3ead93673337ca6569cc9c81be44dbcbd"
p = DataProcessor(
    data_source="tushare",
    start_date=TRAIN_START_DATE,
    end_date=TRADE_END_DATE,
    time_interval=TIME_INTERVAL,
    **kwargs,
)


# download and clean
p.download_data(ticker_list=ticker_list)
p.clean_data()
p.fillna()

# add_technical_indicator
p.add_technical_indicator(config.INDICATORS)
p.fillna()
print(f"p.dataframe: {p.dataframe}")

# -------- 改为读取本地数据文件 ----------
import pickle

# with open(rf'/Users/qinchaoyi/workspace/Ultron/code/AI4Fin/FinRL-Tutorials/1-Introduction/datasets/p.pkl', 'wb') as f:
#     pickle.dump(p, f)

# with open(rf'/Users/qinchaoyi/workspace/Ultron/code/AI4Fin/FinRL-Tutorials/1-Introduction/datasets/p.pkl', 'rb') as f:
#     p = pickle.load(f)
#     print(f"p.dataframe: {p.dataframe}")


### Split traning dataset

train = p.data_split(p.dataframe, TRAIN_START_DATE, TRAIN_END_DATE)
print(f"len(train.tic.unique()): {len(train.tic.unique())}")

print(f"train.tic.unique(): {train.tic.unique()}")

print(f"train.head(): {train.head()}")

print(f"train.shape: {train.shape}")

stock_dimension = len(train.tic.unique())
state_space = stock_dimension * (len(config.INDICATORS) + 2) + 1
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

### Train

env_kwargs = {
    "stock_dim": stock_dimension,
    "hmax": 100000,
    "initial_amount": 1000000,
    "buy_cost_pct": 1.25e-3,
    "sell_cost_pct": 1.25e-3,
    "reward_scaling": 1e-3,
    "state_space": state_space,
    "action_space": stock_dimension,
    "tech_indicator_list": config.INDICATORS,
    "print_verbosity": 1,
    "initial_buy": True,
    "hundred_each_trade": True,
}

e_train_gym = StockTradingEnv(df=train, **env_kwargs)

## DDPG

env_train, _ = e_train_gym.get_sb_env()
print(f"print(type(env_train)): {print(type(env_train))}")

# agent = DRLAgent(env=env_train)
# DDPG_PARAMS = {
#     "batch_size": 128,
#     "buffer_size": 50000,
#     "learning_rate": 0.0005,
#     "action_noise": "normal",
#     "gamma": 0.95,
#     "tau": 0.01,
# }
# POLICY_KWARGS = dict(net_arch=dict(pi=[64, 64], qf=[400, 300]))
# model_ddpg = agent.get_model(
#     "ddpg", model_kwargs=DDPG_PARAMS, policy_kwargs=POLICY_KWARGS
# )

# trained_ddpg = agent.train_model(
#     model=model_ddpg, tb_log_name="ddpg", total_timesteps=10000
# )

## 也尝试PPO算法 - 对单股票可能更适合
agent_ppo = DRLAgent(env=env_train)

PPO_PARAMS = {
    "learning_rate": 0.0003,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.95,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,  # 增加熵系数鼓励探索
}

model_ppo = agent_ppo.get_model("ppo", model_kwargs=PPO_PARAMS)

trained_ppo = agent_ppo.train_model(
    model=model_ppo, tb_log_name="ppo_single_stock", total_timesteps=100000
)

# ## A2C

# agent = DRLAgent(env=env_train)
# model_a2c = agent.get_model("a2c")

# trained_a2c = agent.train_model(
#     model=model_a2c, tb_log_name="a2c", total_timesteps=50000
# )

### Trade

trade = p.data_split(p.dataframe, TRADE_START_DATE, TRADE_END_DATE)
env_kwargs = {
    "stock_dim": stock_dimension,
    "hmax": 10000,
    "initial_amount": 1000000,
    "buy_cost_pct": 1.25e-3,
    "sell_cost_pct": 1.25e-3,
    "reward_scaling": 1e-3,
    "state_space": state_space,
    "action_space": stock_dimension,
    "tech_indicator_list": config.INDICATORS,
    "print_verbosity": 1,
    "initial_buy": True,
    "hundred_each_trade": True,
}
e_trade_gym = StockTradingEnv(df=trade, **env_kwargs)

df_account_value, df_actions = DRLAgent.DRL_prediction(
    model=trained_ppo, environment=e_trade_gym
)

df_actions.to_csv("action.csv", index=False)
print(f"df_actions: {df_actions}")

result_dir_path = rf'/Users/qinchaoyi/workspace/Ultron/code/AI4Fin/FinRL-Tutorials/1-Introduction/results'
result_data = {
    'df_account_value': df_account_value,
    'trade': trade,
}

# ### Backtest
# with open(rf'{result_dir_path}/result_data.pkl', 'wb') as f:
#     pickle.dump(result_data, f)
# with open(rf'{result_dir_path}/result_data.pkl', 'rb') as f:
#     result_data = pickle.load(f)

#     df_account_value = result_data['df_account_value']
#     trade = result_data['trade']


# # matplotlib inline
# plotter = ReturnPlotter(df_account_value, trade, TRADE_START_DATE, TRADE_END_DATE)
# # plotter.plot_all()

# plotter.plot()

# # matplotlib inline
# # # ticket: SSE 50：000016
# # plotter.plot("000016")

# #### Use pyfolio

# # CSI 300
# baseline_df = plotter.get_baseline("399300")


# daily_return = plotter.get_return(df_account_value)
# daily_return_base = plotter.get_return(baseline_df, value_col_name="close")

# perf_func = timeseries.perf_stats
# perf_stats_all = perf_func(
#     returns=daily_return,
#     factor_returns=daily_return_base,
#     positions=None,
#     transactions=None,
#     turnover_denom="AGB",
# )
# print("==============DRL Strategy Stats===========")
# print(f"perf_stats_all: {perf_stats_all}")


# daily_return = plotter.get_return(df_account_value)
# daily_return_base = plotter.get_return(baseline_df, value_col_name="close")

# perf_func = timeseries.perf_stats
# perf_stats_all = perf_func(
#     returns=daily_return_base,
#     factor_returns=daily_return_base,
#     positions=None,
#     transactions=None,
#     turnover_denom="AGB",
# )
# print("==============Baseline Strategy Stats===========")

# print(f"perf_stats_all: {perf_stats_all}")
