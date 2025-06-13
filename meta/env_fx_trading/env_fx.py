import datetime
import math
import random

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
from stable_baselines3.common.vec_env import DummyVecEnv

from meta.env_fx_trading.util.log_render import render_to_file
from meta.env_fx_trading.util.plot_chart import TradingChart
from meta.env_fx_trading.util.read_config import EnvConfig


class tgym(gym.Env):
    """
    外汇交易环境 - Forex Trading Environment
    
    交易模式 (Trading Mode):
    - 多货币对同时交易 - Trading multiple currency pairs simultaneously
    - 支持买入(0)、卖出(1)、持有(2)三种动作 - Supports Buy(0), Sell(1), Hold(2) actions
    - 基于止损止盈机制进行风险管理 - Risk management based on stop-loss and take-profit mechanisms
    - 支持限价单和市价单 - Supports limit orders and market orders
    
    应用场景 (Application Scenarios):
    - 外汇市场的短期交易策略 - Short-term trading strategies in forex markets
    - 多货币对套利策略 - Multi-currency arbitrage strategies
    - 高频交易和算法交易 - High-frequency and algorithmic trading
    - 外汇风险对冲 - Forex risk hedging
    
    核心特点 (Key Features):
    - 动作空间: 每个货币对的三元动作(买/卖/持有) - Action space: ternary actions (buy/sell/hold) for each pair
    - 状态空间: OHLC价格数据 + 技术指标 + 持仓信息 - State space: OHLC price data + technical indicators + position info
    - 奖励函数: 基于点数(Point)的盈亏计算 - Reward function: profit/loss calculation based on points
    - 风险控制: 支持隔夜持仓费用和最大回撤限制 - Risk control: overnight fees and maximum drawdown limits
    """
    """forex/future/option trading gym environment # 外汇/期货/期权交易gym环境
    1. Three action space (0 Buy, 1 Sell, 2 Nothing) # 三种动作空间(0买入，1卖出，2不操作)
    2. Multiple trading pairs (EURUSD, GBPUSD...) under same time frame # 同一时间框架下的多个交易对(EURUSD, GBPUSD...)
    3. Timeframe from 1 min to daily as long as use candlestick bar (Open, High, Low, Close) # 时间框架从1分钟到日线，只要使用K线数据(开高低收)
    4. Use StopLose, ProfitTaken to realize rewards. each pair can configure it own SL and PT in configure file # 使用止损止盈来实现奖励。每个交易对可以在配置文件中设置自己的止损和止盈
    5. Configure over night cash penalty and each pair's transaction fee and overnight position holding penalty # 配置隔夜现金惩罚、每个交易对的交易费用和隔夜持仓惩罚
    6. Split dataset into daily, weekly or monthly..., with fixed time steps, at end of len(df). The business # 将数据集分割为日、周或月...，具有固定时间步长，在len(df)结束时。业务
        logic will force to Close all positions at last Close price (game over). # 逻辑将强制以最后收盘价平仓所有持仓(游戏结束)
    7. Must have df column name: [(time_col),(asset_col), Open,Close,High,Low,day] (case sensitive) # 必须有df列名：[(时间列),(资产列), Open,Close,High,Low,day](区分大小写)
    8. Addition indicators can add during the data process. 78 available TA indicator from Finta # 可以在数据处理过程中添加其他指标。Finta提供78个可用的技术分析指标
    9. Customized observation list handled in json config file. # 自定义观察列表在json配置文件中处理
    10. ProfitTaken = fraction_action * max_profit_taken + SL. # 止盈 = 分数动作 * 最大止盈 + 止损
    11. SL is pre-fixed # 止损是预先固定的
    12. Limit order can be configure, if limit_order == True, the action will preset buy or sell at Low or High of the bar, # 可以配置限价单，如果limit_order == True，动作将预设在K线的最低价或最高价买入或卖出
        with a limit_order_expiration (n bars). It will be triggered if the price go cross. otherwise, it will be drop off # 具有限价单过期时间(n个K线)。如果价格穿越将被触发，否则将被删除
    13. render mode: # 渲染模式：
        human -- display each steps realized reward on console # human -- 在控制台显示每步实现的奖励
        file -- create a transaction log # file -- 创建交易日志
        graph -- create transaction in graph (under development) # graph -- 在图表中创建交易(开发中)
    14.
    15. Reward, we want to incentivize profit that is sustained over long periods of time. # 奖励，我们希望激励长期持续的利润
        At each step, we will set the reward to the account balance multiplied by # 在每一步，我们将奖励设置为账户余额乘以
        some fraction of the number of time steps so far.The purpose of this is to delay # 到目前为止时间步数的某个分数。这样做的目的是延迟
        rewarding the agent too fast in the early stages and allow it to explore # 在早期阶段过快奖励智能体，让它充分探索
        sufficiently before optimizing a single strategy too deeply. # 在过度优化单一策略之前
        It will also reward agents that maintain a higher balance for longer, # 它还会奖励那些长期保持较高余额的智能体
        rather than those who rapidly gain money using unsustainable strategies. # 而不是那些使用不可持续策略快速赚钱的智能体
    16. Observation_space contains all of the input variables we want our agent # 观察空间包含我们希望智能体
        to consider before making, or not making a trade. We want our agent to "see" # 在进行或不进行交易之前考虑的所有输入变量。我们希望智能体能够"看到"
        the forex data points (Open price, High, Low, Close, time serial, TA) in the game window, # 游戏窗口中的外汇数据点(开盘价、最高价、最低价、收盘价、时间序列、技术分析)
        as well a couple other data points like its account balance, current positions, # 以及其他一些数据点，如账户余额、当前持仓
        and current profit.The intuition here is that for each time step, we want our agent # 和当前利润。这里的直觉是，对于每个时间步，我们希望智能体
        to consider the price action leading up to the current price, as well as their # 考虑导致当前价格的价格行为，以及它们
        own portfolio's status in order to make an informed decision for the next action. # 自己投资组合的状态，以便为下一个动作做出明智的决定
    17. reward is forex trading unit Point, it can be configure for each trading pair # 奖励是外汇交易单位点，可以为每个交易对配置
    """

    metadata = {"render.modes": ["graph", "human", "file", "none"]}

    def __init__(
        self,
        df,  # 市场数据框 - Market dataframe
        env_config_file="./neo_finrl/env_fx_trading/config/gdbusd-test-1.json",  # 环境配置文件 - Environment config file
    ) -> None:
        assert df.ndim == 2
        super(tgym, self).__init__()
        self.cf = EnvConfig(env_config_file)  # 配置管理器 - Configuration manager
        self.observation_list = self.cf.env_parameters("observation_list")  # 观察列表 - Observation list

        self.balance_initial = self.cf.env_parameters("balance")  # 初始余额 - Initial balance
        self.over_night_cash_penalty = self.cf.env_parameters("over_night_cash_penalty")  # 隔夜现金惩罚 - Overnight cash penalty
        self.asset_col = self.cf.env_parameters("asset_col")  # 资产列 - Asset column
        self.time_col = self.cf.env_parameters("time_col")  # 时间列 - Time column
        self.random_start = self.cf.env_parameters("random_start")  # 随机开始 - Random start
        self.log_filename = (
            self.cf.env_parameters("log_filename")
            + datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            + ".csv"
        )  # 日志文件名 - Log filename

        self.df = df
        self.df["_time"] = df[self.time_col]  # 内部时间列 - Internal time column
        self.df["_day"] = df["weekday"]  # 星期列 - Weekday column
        self.assets = df[self.asset_col].unique()  # 资产列表 - Asset list
        self.dt_datetime = df[self.time_col].sort_values().unique()  # 日期时间列表 - Datetime list
        self.df = self.df.set_index(self.time_col)
        self.visualization = False  # 是否可视化 - Whether to visualize

        # --- reset value --- # 重置值
        self.equity_list = [0] * len(self.assets)  # 权益列表 - Equity list
        self.balance = self.balance_initial  # 当前余额 - Current balance
        self.total_equity = self.balance + sum(self.equity_list)  # 总权益 - Total equity
        self.ticket_id = 0  # 交易单号 - Trade ticket ID
        self.transaction_live = []  # 活跃交易 - Live transactions
        self.transaction_history = []  # 交易历史 - Transaction history
        self.transaction_limit_order = []  # 限价单 - Limit orders
        self.current_draw_downs = [0.0] * len(self.assets)  # 当前回撤 - Current drawdowns
        self.max_draw_downs = [0.0] * len(self.assets)  # 最大回撤 - Maximum drawdowns
        self.max_draw_down_pct = sum(self.max_draw_downs) / self.balance * 100  # 最大回撤百分比 - Max drawdown percentage
        self.current_step = 0  # 当前步数 - Current step
        self.episode = -1  # 回合数 - Episode number
        self.current_holding = [0] * len(self.assets)  # 当前持仓 - Current holdings
        self.tranaction_open_this_step = []  # 本步开仓交易 - Transactions opened this step
        self.tranaction_close_this_step = []  # 本步平仓交易 - Transactions closed this step
        self.current_day = 0  # 当前日期 - Current day
        self.done_information = ""  # 完成信息 - Done information
        self.log_header = True  # 日志头 - Log header
        # --- end reset --- # 重置结束
        self.cached_data = [
            self.get_observation_vector(_dt) for _dt in self.dt_datetime
        ]  # 缓存数据 - Cached data
        self.cached_time_serial = (
            (self.df[["_time", "_day"]].sort_values("_time")).drop_duplicates()
        ).values.tolist()  # 缓存时间序列 - Cached time series

        self.reward_range = (-np.inf, np.inf)  # 奖励范围 - Reward range
        self.action_space = spaces.Box(low=0, high=3, shape=(len(self.assets),))  # 动作空间：每个资产0-3 - Action space: 0-3 for each asset
        # first two 3 = balance,current_holding, max_draw_down_pct # 前三个 = 余额,当前持仓, 最大回撤百分比
        _space = 3 + len(self.assets) + len(self.assets) * len(self.observation_list)  # 观察空间维度 - Observation space dimension
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(_space,))
        print(
            f"initial done:\n"  # 初始化完成：
            f"observation_list:{self.observation_list}\n "  # 观察列表：
            f"assets:{self.assets}\n "  # 资产：
            f"time serial: {min(self.dt_datetime)} -> {max(self.dt_datetime)} length: {len(self.dt_datetime)}"  # 时间序列：
        )
        self._seed()

    def _seed(self, seed=None):
        """设置随机种子 - Set random seed"""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _history_df(self, i):
        pass

    def _take_action(self, actions, done):
        """执行交易动作 - Execute trading actions"""
        # action = math.floor(x), # 动作 = math.floor(x)
        # profit_taken = math.ceil((x- math.floor(x)) * profit_taken_max - stop_loss_max ) # 止盈 = math.ceil((x- math.floor(x)) * 最大止盈 - 最大止损)
        # _actions = np.floor(actions).astype(int)
        # _profit_takens = np.ceil((actions - np.floor(actions)) *self.cf.symbol(self.assets[i],"profit_taken_max")).astype(int)
        _action = 2  # 默认动作：持有 - Default action: hold
        _profit_taken = 0  # 止盈 - Profit taken
        rewards = [0] * len(self.assets)  # 奖励列表 - Rewards list
        self.tranaction_open_this_step = []  # 本步开仓列表 - Open transactions this step
        self.tranaction_close_this_step = []  # 本步平仓列表 - Close transactions this step
        # need use multiply assets # 需要使用多个资产
        for i, x in enumerate(actions):
            self._o = self.get_observation(self.current_step, i, "Open")  # 开盘价 - Open price
            self._h = self.get_observation(self.current_step, i, "High")  # 最高价 - High price
            self._l = self.get_observation(self.current_step, i, "Low")   # 最低价 - Low price
            self._c = self.get_observation(self.current_step, i, "Close") # 收盘价 - Close price
            self._t = self.get_observation(self.current_step, i, "_time") # 时间 - Time
            self._day = self.get_observation(self.current_step, i, "_day") # 星期 - Weekday
            _action = math.floor(x)  # 获取动作类型 - Get action type
            rewards[i] = self._calculate_reward(i, done)  # 计算奖励 - Calculate reward
            if self.cf.symbol(self.assets[i], "limit_order"):  # 如果支持限价单 - If limit order is supported
                self._limit_order_process(i, _action, done)
            if (
                _action in (0, 1)  # 如果是买入或卖出动作 - If buy or sell action
                and not done  # 且游戏未结束 - And game not done
                and self.current_holding[i]  # 且当前持仓数量 - And current holding
                < self.cf.symbol(self.assets[i], "max_current_holding")  # 小于最大持仓限制 - Less than max holding limit
            ):
                # generating PT based on action fraction # 基于动作分数生成止盈
                _profit_taken = math.ceil(
                    (x - _action) * self.cf.symbol(self.assets[i], "profit_taken_max")
                ) + self.cf.symbol(self.assets[i], "stop_loss_max")
                self.ticket_id += 1  # 交易单号递增 - Increment ticket ID
                if self.cf.symbol(self.assets[i], "limit_order"):  # 限价单处理 - Limit order processing
                    transaction = {
                        "Ticket": self.ticket_id,  # 交易单号 - Ticket ID
                        "Symbol": self.assets[i],  # 交易品种 - Trading symbol
                        "ActionTime": self._t,     # 动作时间 - Action time
                        "Type": _action,           # 动作类型 - Action type
                        "Lot": 1,                  # 手数 - Lot size
                        "ActionPrice": self._l if _action == 0 else self._h,  # 动作价格 - Action price
                        "SL": self.cf.symbol(self.assets[i], "stop_loss_max"),  # 止损 - Stop loss
                        "PT": _profit_taken,       # 止盈 - Profit taken
                        "MaxDD": 0,                # 最大回撤 - Max drawdown
                        "Swap": 0.0,               # 库存费 - Swap fee
                        "CloseTime": "",           # 平仓时间 - Close time
                        "ClosePrice": 0.0,         # 平仓价格 - Close price
                        "Point": 0,                # 点数 - Points
                        "Reward": -self.cf.symbol(self.assets[i], "transaction_fee"),  # 奖励(扣除手续费) - Reward (minus transaction fee)
                        "DateDuration": self._day, # 持续天数 - Duration in days
                        "Status": 0,               # 状态 - Status
                        "LimitStep": self.current_step,  # 限价步数 - Limit step
                        "ActionStep": -1,          # 动作步数 - Action step
                        "CloseStep": -1,           # 平仓步数 - Close step
                    }
                    self.transaction_limit_order.append(transaction)
                else:  # 市价单处理 - Market order processing
                    transaction = {
                        "Ticket": self.ticket_id,
                        "Symbol": self.assets[i],
                        "ActionTime": self._t,
                        "Type": _action,
                        "Lot": 1,
                        "ActionPrice": self._c,    # 市价执行 - Market price execution
                        "SL": self.cf.symbol(self.assets[i], "stop_loss_max"),
                        "PT": _profit_taken,
                        "MaxDD": 0,
                        "Swap": 0.0,
                        "CloseTime": "",
                        "ClosePrice": 0.0,
                        "Point": 0,
                        "Reward": -self.cf.symbol(self.assets[i], "transaction_fee"),
                        "DateDuration": self._day,
                        "Status": 0,
                        "LimitStep": self.current_step,
                        "ActionStep": self.current_step,
                        "CloseStep": -1,
                    }
                    self.current_holding[i] += 1
                    self.tranaction_open_this_step.append(transaction)
                    self.balance -= self.cf.symbol(self.assets[i], "transaction_fee")
                    self.transaction_live.append(transaction)

        return sum(rewards)

    def _calculate_reward(self, i, done):
        _total_reward = 0
        _max_draw_down = 0
        for tr in self.transaction_live:
            if tr["Symbol"] == self.assets[i]:
                _point = self.cf.symbol(self.assets[i], "point")
                # cash discount overnight
                if self._day > tr["DateDuration"]:
                    tr["DateDuration"] = self._day
                    tr["Reward"] -= self.cf.symbol(self.assets[i], "over_night_penalty")

                if tr["Type"] == 0:  # buy
                    # stop loss trigger
                    _sl_price = tr["ActionPrice"] - tr["SL"] / _point
                    _pt_price = tr["ActionPrice"] + tr["PT"] / _point
                    if done:
                        p = (self._c - tr["ActionPrice"]) * _point
                        self._manage_tranaction(tr, p, self._c, status=2)
                        _total_reward += p
                    elif self._l <= _sl_price:
                        self._manage_tranaction(tr, -tr["SL"], _sl_price)
                        _total_reward += -tr["SL"]
                        self.current_holding[i] -= 1
                    elif self._h >= _pt_price:
                        self._manage_tranaction(tr, tr["PT"], _pt_price)
                        _total_reward += tr["PT"]
                        self.current_holding[i] -= 1
                    else:  # still open
                        self.current_draw_downs[i] = int(
                            (self._l - tr["ActionPrice"]) * _point
                        )
                        _max_draw_down += self.current_draw_downs[i]
                        if (
                            self.current_draw_downs[i] < 0
                            and tr["MaxDD"] > self.current_draw_downs[i]
                        ):
                            tr["MaxDD"] = self.current_draw_downs[i]

                elif tr["Type"] == 1:  # sell
                    # stop loss trigger
                    _sl_price = tr["ActionPrice"] + tr["SL"] / _point
                    _pt_price = tr["ActionPrice"] - tr["PT"] / _point
                    if done:
                        p = (tr["ActionPrice"] - self._c) * _point
                        self._manage_tranaction(tr, p, self._c, status=2)
                        _total_reward += p
                    elif self._h >= _sl_price:
                        self._manage_tranaction(tr, -tr["SL"], _sl_price)
                        _total_reward += -tr["SL"]
                        self.current_holding[i] -= 1
                    elif self._l <= _pt_price:
                        self._manage_tranaction(tr, tr["PT"], _pt_price)
                        _total_reward += tr["PT"]
                        self.current_holding[i] -= 1
                    else:
                        self.current_draw_downs[i] = int(
                            (tr["ActionPrice"] - self._h) * _point
                        )
                        _max_draw_down += self.current_draw_downs[i]
                        if (
                            self.current_draw_downs[i] < 0
                            and tr["MaxDD"] > self.current_draw_downs[i]
                        ):
                            tr["MaxDD"] = self.current_draw_downs[i]

                if _max_draw_down > self.max_draw_downs[i]:
                    self.max_draw_downs[i] = _max_draw_down

        return _total_reward

    def _limit_order_process(self, i, _action, done):
        for tr in self.transaction_limit_order:
            if tr["Symbol"] == self.assets[i]:
                if tr["Type"] != _action or done:
                    self.transaction_limit_order.remove(tr)
                    tr["Status"] = 3
                    tr["CloseStep"] = self.current_step
                    self.transaction_history.append(tr)
                elif (tr["ActionPrice"] >= self._l and _action == 0) or (
                    tr["ActionPrice"] <= self._h and _action == 1
                ):
                    tr["ActionStep"] = self.current_step
                    self.current_holding[i] += 1
                    self.balance -= self.cf.symbol(self.assets[i], "transaction_fee")
                    self.transaction_limit_order.remove(tr)
                    self.transaction_live.append(tr)
                    self.tranaction_open_this_step.append(tr)
                elif (
                    tr["LimitStep"]
                    + self.cf.symbol(self.assets[i], "limit_order_expiration")
                    > self.current_step
                ):
                    tr["CloseStep"] = self.current_step
                    tr["Status"] = 4
                    self.transaction_limit_order.remove(tr)
                    self.transaction_history.append(tr)

    def _manage_tranaction(self, tr, _p, close_price, status=1):
        self.transaction_live.remove(tr)
        tr["ClosePrice"] = close_price
        tr["Point"] = int(_p)
        tr["Reward"] = int(tr["Reward"] + _p)
        tr["Status"] = status
        tr["CloseTime"] = self._t
        self.balance += int(tr["Reward"])
        self.total_equity -= int(abs(tr["Reward"]))
        self.tranaction_close_this_step.append(tr)
        self.transaction_history.append(tr)

    def step(self, actions):
        # Execute one time step within the environment
        self.current_step += 1
        done = self.balance <= 0 or self.current_step == len(self.dt_datetime) - 1
        if done:
            self.done_information += f"Episode: {self.episode} Balance: {self.balance} Step: {self.current_step}\n"
            self.visualization = True
        reward = self._take_action(actions, done)
        if self._day > self.current_day:
            self.current_day = self._day
            self.balance -= self.over_night_cash_penalty
        if self.balance != 0:
            self.max_draw_down_pct = abs(sum(self.max_draw_downs) / self.balance * 100)

            # no action anymore
        obs = (
            [self.balance, self.max_draw_down_pct]
            + self.current_holding
            + self.current_draw_downs
            + self.get_observation(self.current_step)
        )
        return (
            np.array(obs).astype(np.float32),
            reward,
            done,
            {"Close": self.tranaction_close_this_step},
        )

    def get_observation(self, _step, _iter=0, col=None):
        if col is None:
            return self.cached_data[_step]
        if col == "_day":
            return self.cached_time_serial[_step][1]

        elif col == "_time":
            return self.cached_time_serial[_step][0]
        col_pos = -1
        for i, _symbol in enumerate(self.observation_list):
            if _symbol == col:
                col_pos = i
                break
        assert col_pos >= 0
        return self.cached_data[_step][_iter * len(self.observation_list) + col_pos]

    def get_observation_vector(self, _dt, cols=None):
        cols = self.observation_list
        v = []
        for a in self.assets:
            subset = self.df.query(
                f'{self.asset_col} == "{a}" & {self.time_col} == "{_dt}"'
            )
            assert not subset.empty
            v += subset.loc[_dt, cols].tolist()
        assert len(v) == len(self.assets) * len(cols)
        return v

    def reset(self):
        # Reset the state of the environment to an initial state
        self.seed()

        if self.random_start:
            self.current_step = random.choice(range(int(len(self.dt_datetime) * 0.5)))
        else:
            self.current_step = 0

        self.equity_list = [0] * len(self.assets)
        self.balance = self.balance_initial
        self.total_equity = self.balance + sum(self.equity_list)
        self.ticket_id = 0
        self.transaction_live = []
        self.transaction_history = []
        self.transaction_limit_order = []
        self.current_draw_downs = [0.0] * len(self.assets)
        self.max_draw_downs = [0.0] * len(self.assets)
        self.max_draw_down_pct = sum(self.max_draw_downs) / self.balance * 100
        self.episode = -1
        self.current_holding = [0] * len(self.assets)
        self.tranaction_open_this_step = []
        self.tranaction_close_this_step = []
        self.current_day = 0
        self.done_information = ""
        self.log_header = True
        self.visualization = False

        _space = (
            [self.balance, self.max_draw_down_pct]
            + [0] * len(self.assets)
            + [0] * len(self.assets)
            + self.get_observation(self.current_step)
        )
        return np.array(_space).astype(np.float32)

    def render(self, mode="human", title=None, **kwargs):
        # Render the environment to the screen
        if mode in ("human", "file"):
            printout = mode == "human"
            pm = {
                "log_header": self.log_header,
                "log_filename": self.log_filename,
                "printout": printout,
                "balance": self.balance,
                "balance_initial": self.balance_initial,
                "tranaction_close_this_step": self.tranaction_close_this_step,
                "done_information": self.done_information,
            }
            render_to_file(**pm)
            if self.log_header:
                self.log_header = False
        elif mode == "graph" and self.visualization:
            print("plotting...")
            p = TradingChart(self.df, self.transaction_history)
            p.plot()

    def close(self):
        pass

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs
