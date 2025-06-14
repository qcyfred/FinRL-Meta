"""From FinRL https://github.com/AI4Finance-LLC/FinRL/tree/master/finrl/env"""

import math

import gym
import matplotlib
import numpy as np
import pandas as pd
from gym import spaces
from gym.utils import seeding

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv
from pathlib import Path

try:
    import quantstats as qs
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        """QuantStats module not found, environment can't plot results and calculate indicadors.
        This module is not installed with FinRL. Install by running one of the options:
        pip install quantstats --upgrade --no-cache-dir
        conda install -c ranaroussi quantstats
        """
    )


class PortfolioOptimizationEnv(gym.Env):
    """
    投资组合权重优化环境 - Portfolio Weight Optimization Environment
    
    交易模式 (Trading Mode):
    - 决定各个资产在投资组合中的权重分配 - Determines weight allocation of each asset in the portfolio
    - 权重总和必须等于1，包括现金权重 - Total weights must equal 1, including cash weight
    - 基于历史数据动态调整投资组合配置 - Dynamically adjusts portfolio allocation based on historical data
    
    应用场景 (Application Scenarios):
    - 资产配置策略研究 - Asset allocation strategy research
    - 机构投资者的投资组合管理 - Portfolio management for institutional investors
    - 风险平价和因子投资策略 - Risk parity and factor investing strategies
    - 多资产类别的动态配置 - Dynamic allocation across multiple asset classes
    
    核心特点 (Key Features):
    - 动作空间: 归一化权重向量(总和为1) - Action space: normalized weight vector (sum to 1)
    - 状态空间: 多维时间序列数据(价格、技术指标等) - State space: multi-dimensional time series data
    - 奖励函数: 基于投资组合收益率 - Reward function: based on portfolio returns
    - 支持多种数据归一化方法 - Supports various data normalization methods
    """
    """A portfolio allocantion environment for OpenAI gym. # 用于OpenAI gym的投资组合分配环境

    This environment simulates the interactions between an agent and the financial market # 该环境模拟智能体与金融市场之间的交互
    based on data provided by a dataframe. The dataframe contains the time series of # 基于数据框提供的数据。数据框包含时间序列
    features defined by the user (such as closing, high and low prices) and must have # 用户定义的特征(如收盘价、最高价和最低价)，必须包含
    a time and a tic column with a list of datetimes and ticker symbols respectively. # 时间列和股票代码列，分别包含日期时间和股票代码列表
    An example of dataframe is shown below:: # 数据框示例如下::

            date        high            low             close           tic
        0   2020-12-23  0.157414        0.127420        0.136394        ADA-USD
        1   2020-12-23  34.381519       30.074295       31.097898       BNB-USD
        2   2020-12-23  24024.490234    22802.646484    23241.345703    BTC-USD
        3   2020-12-23  0.004735        0.003640        0.003768        DOGE-USD
        4   2020-12-23  637.122803      560.364258      583.714600      ETH-USD
        ... ...         ...             ...             ...             ...

    Based on this dataframe, the environment will create an observation space that can # 基于此数据框，环境将创建一个观察空间，可以是
    be a Dict or a Box. The Box observation space is a three-dimensional array of shape # Dict或Box。Box观察空间是形状为(f, n, t)的三维数组
    (f, n, t), where f is the number of features, n is the number of stocks in the # 其中f是特征数量，n是投资组合中的股票数量
    portfolio and t is the user-defined time window. If the environment is created with # t是用户定义的时间窗口。如果环境创建时
    the parameter return_last_action set to True, the observation space is a Dict with # 参数return_last_action设置为True，观察空间是一个Dict，包含
    the following keys:: # 以下键::

        {
        "state": three-dimensional Box (f, n, t) representing the time series, # "state": 表示时间序列的三维Box (f, n, t)
        "last_action": one-dimensional Box (n+1,) representing the portfolio weights # "last_action": 表示投资组合权重的一维Box (n+1,)
        }

    Note that the action space of this environment is an one-dimensional Box with size # 注意此环境的动作空间是大小为n+1的一维Box
    n + 1 because the portfolio weights must contains the weights related to all the # 因为投资组合权重必须包含所有
    stocks in the portfolio and to the remaining cash. # 投资组合中股票的权重以及剩余现金的权重

    Attributes:
        action_space: Action space. # 动作空间
        observation_space: Observation space. # 观察空间
        episode_length: Number of timesteps of an episode. # 一个回合的时间步数
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df,  # 市场数据框 - Market dataframe
        initial_amount,  # 初始投资金额 - Initial investment amount
        order_df=True,  # 是否按时间排序数据框 - Whether to order dataframe by time
        return_last_action=False,  # 是否返回上一步动作 - Whether to return last action
        normalize_df="by_previous_time",  # 数据归一化方法 - Data normalization method
        reward_scaling=1,  # 奖励缩放因子 - Reward scaling factor
        comission_fee_model="trf",  # 手续费模型 - Commission fee model
        comission_fee_pct=0,  # 手续费百分比 - Commission fee percentage
        features=["close", "high", "low"],  # 使用的特征列表 - List of features to use
        valuation_feature="close",  # 估值特征 - Valuation feature
        time_column="date",  # 时间列名 - Time column name
        time_format="%Y-%m-%d",  # 时间格式 - Time format
        tic_column="tic",  # 股票代码列名 - Ticker column name
        time_window=1,  # 时间窗口大小 - Time window size
        cwd="./",  # 当前工作目录 - Current working directory
        new_gym_api=False,  # 是否使用新的gym API - Whether to use new gym API
    ):
        """Initializes environment's instance. # 初始化环境实例

        Args:
            df: Dataframe with market information over a period of time. # 包含一段时间内市场信息的数据框
            initial_amount: Initial amount of cash available to be invested. # 可用于投资的初始现金金额
            order_df: If True input dataframe is ordered by time. # 如果为True，输入数据框按时间排序
            return_last_action: If True, observations also return the last performed # 如果为True，观察还返回最后执行的
                action. Note that, in that case, the observation space is a Dict. # 动作。注意，在这种情况下，观察空间是Dict
            normalize_df: Defines the normalization method applied to input dataframe. # 定义应用于输入数据框的归一化方法
                Possible values are "by_previous_time", "by_fist_time_window_value", # 可能的值有"by_previous_time", "by_fist_time_window_value"
                "by_COLUMN_NAME" (where COLUMN_NAME must be changed to a real column # "by_COLUMN_NAME"(其中COLUMN_NAME必须改为真实列名)
                name) and a custom function. If None no normalization is done. # 和自定义函数。如果为None则不进行归一化
            reward_scaling: A scaling factor to multiply the reward function. This # 奖励函数的缩放因子。这个
                factor can help training. # 因子可以帮助训练
            comission_fee_model: Model used to simulate comission fee. Possible values # 用于模拟手续费的模型。可能的值
                are "trf" (for transaction remainder factor model) and "wvm" (for weights # 是"trf"(交易余额因子模型)和"wvm"(权重
                vector modifier model). If None, commission fees are not considered. # 向量修改模型)。如果为None，则不考虑手续费
            comission_fee_pct: Percentage to be used in comission fee. It must be a value # 手续费中使用的百分比。必须是
                between 0 and 1. # 0到1之间的值
            features: List of features to be considered in the observation space. The # 观察空间中要考虑的特征列表。列表中的
                items # 项目
                of the list must be names of columns of the input dataframe. # 必须是输入数据框的列名
            valuation_feature: Feature to be considered in the portfolio value calculation. # 投资组合价值计算中要考虑的特征
            time_column: Name of the dataframe's column that contain the datetimes that # 包含索引数据框的日期时间的
                index the dataframe. # 数据框列的名称
            time_format: Formatting string of time column. # 时间列的格式字符串
            tic_name: Name of the dataframe's column that contain ticker symbols. # 包含股票代码的数据框列的名称
            time_window: Size of time window. # 时间窗口的大小
            cwd: Local repository in which resulting graphs will be saved. # 保存结果图表的本地存储库
            new_gym_api: If True, the environment will use the new gym api standard for # 如果为True，环境将使用新的gym api标准
                step and reset methods. # 用于step和reset方法
        """
        # super(StockEnv, self).__init__()
        # money = 10 , scope = 1
        self._time_window = time_window  # 时间窗口 - Time window
        self._time_index = time_window - 1  # 时间索引 - Time index
        self._time_column = time_column  # 时间列 - Time column
        self._time_format = time_format  # 时间格式 - Time format
        self._tic_column = tic_column  # 股票代码列 - Ticker column
        self._df = df  # 数据框 - Dataframe
        self._initial_amount = initial_amount  # 初始金额 - Initial amount
        self._return_last_action = return_last_action  # 是否返回最后动作 - Whether to return last action
        self._reward_scaling = reward_scaling  # 奖励缩放 - Reward scaling
        self._comission_fee_pct = comission_fee_pct  # 手续费百分比 - Commission fee percentage
        self._comission_fee_model = comission_fee_model  # 手续费模型 - Commission fee model
        self._features = features  # 特征列表 - Features list
        self._valuation_feature = valuation_feature  # 估值特征 - Valuation feature
        self._cwd = Path(cwd)  # 工作目录 - Working directory
        self._new_gym_api = new_gym_api  # 新gym API - New gym API

        # results file # 结果文件
        self._results_file = self._cwd / "results" / "rl"
        self._results_file.mkdir(parents=True, exist_ok=True)

        # price variation # 价格变化
        self._df_price_variation = None

        # preprocess data # 预处理数据
        self._preprocess_data(order_df, normalize_df)

        # dims and spaces # 维度和空间
        self._tic_list = self._df[self._tic_column].unique()  # 股票代码列表 - Ticker list
        self._stock_dim = len(self._tic_list)  # 股票维度 - Stock dimension
        action_space = 1 + self._stock_dim  # 动作空间维度(股票数量+现金) - Action space dimension (number of stocks + cash)

        # sort datetimes and define episode length # 排序日期时间并定义回合长度
        self._sorted_times = sorted(set(self._df[time_column]))
        self.episode_length = len(self._sorted_times) - time_window + 1

        # define action space # 定义动作空间
        self.action_space = spaces.Box(low=0, high=1, shape=(action_space,))  # 权重范围[0,1] - Weight range [0,1]

        # define observation state # 定义观察状态
        if self._return_last_action:
            # if  last action must be returned, a dict observation # 如果必须返回最后动作，使用字典观察
            # is defined # 空间
            self.observation_space = spaces.Dict(
                {
                    "state": spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=(len(self._features), self._stock_dim, self._time_window),
                    ),
                    "last_action": spaces.Box(low=0, high=1, shape=(action_space,)),
                }
            )
        else:
            # if information about last action is not relevant, # 如果最后动作信息不相关，
            # a 3D observation space is defined # 定义3D观察空间
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(len(self._features), self._stock_dim, self._time_window),
            )

        self._reset_memory()  # 重置内存 - Reset memory

        self._portfolio_value = self._initial_amount  # 投资组合价值 - Portfolio value
        self._terminal = False  # 是否终止 - Whether terminal

    def step(self, actions):
        """Performs a simulation step.

        Args:
            actions: An unidimensional array containing the new portfolio
                weights.

        Note:
            If the environment was created with "return_last_action" set to
            True, the next state returned will be a Dict. If it's set to False,
            the next state will be a Box. You can check the observation state
            through the attribute "observation_space".

        Returns:
            If "new_gym_api" is set to True, the following tuple is returned:
            (state, reward, terminal, truncated, info). If it's set to False,
            the following tuple is returned: (state, reward, terminal, info).

            state: Next simulation state.
            reward: Reward related to the last performed action.
            terminal: If True, the environment is in a terminal state.
            truncated: If True, the environment has passed it's simulation
                time limit. Currently, it's always False.
            info: A dictionary containing informations about the last state.
        """
        self._terminal = self._time_index >= len(self._sorted_times) - 1

        if self._terminal:
            metrics_df = pd.DataFrame(
                {
                    "date": self._date_memory,
                    "returns": self._portfolio_return_memory,
                    "rewards": self._portfolio_reward_memory,
                    "portfolio_values": self._asset_memory["final"],
                }
            )
            metrics_df.set_index("date", inplace=True)

            plt.plot(metrics_df["portfolio_values"], "r")
            plt.title("Portfolio Value Over Time")
            plt.xlabel("Time")
            plt.ylabel("Portfolio value")
            plt.savefig(self._results_file / "portfolio_value.png")
            plt.close()

            plt.plot(self._portfolio_reward_memory, "r")
            plt.title("Reward Over Time")
            plt.xlabel("Time")
            plt.ylabel("Reward")
            plt.savefig(self._results_file / "reward.png")
            plt.close()

            plt.plot(self._actions_memory)
            plt.title("Actions performed")
            plt.xlabel("Time")
            plt.ylabel("Weight")
            plt.savefig(self._results_file / "actions.png")
            plt.close()

            print("=================================")
            print("Initial portfolio value:{}".format(self._asset_memory["final"][0]))
            print("Final portfolio value: {}".format(self._portfolio_value))
            print(
                "Final accumulative portfolio value: {}".format(
                    self._portfolio_value / self._asset_memory["final"][0]
                )
            )
            print(
                "Maximum DrawDown: {}".format(
                    qs.stats.max_drawdown(metrics_df["portfolio_values"])
                )
            )
            print("Sharpe ratio: {}".format(qs.stats.sharpe(metrics_df["returns"])))
            print("=================================")

            qs.plots.snapshot(
                metrics_df["returns"],
                show=False,
                savefig=self._results_file / "portfolio_summary.png",
            )

            if self._new_gym_api:
                return self._state, self._reward, self._terminal, False, self._info
            return self._state, self._reward, self._terminal, self._info

        else:
            # transform action to numpy array (if it's a list)
            actions = np.array(actions, dtype=np.float32)

            # if necessary, normalize weights
            if math.isclose(np.sum(actions), 1, abs_tol=1e-6) and np.min(actions) >= 0:
                weights = actions
            else:
                weights = self._softmax_normalization(actions)

            # save initial portfolio weights for this time step
            self._actions_memory.append(weights)

            # get last step final weights and portfolio_value
            last_weights = self._final_weights[-1]

            # load next state
            self._time_index += 1
            self._state, self._info = self._get_state_and_info_from_time_index(
                self._time_index
            )

            # if using weights vector modifier, we need to modify weights vector
            if self._comission_fee_model == "wvm":
                delta_weights = weights - last_weights
                delta_assets = delta_weights[1:]  # disconsider
                # calculate fees considering weights modification
                fees = np.sum(np.abs(delta_assets * self._portfolio_value))
                if fees > weights[0] * self._portfolio_value:
                    weights = last_weights
                    # maybe add negative reward
                else:
                    portfolio = weights * self._portfolio_value
                    portfolio[0] -= fees
                    self._portfolio_value = np.sum(portfolio)  # new portfolio value
                    weights = portfolio / self._portfolio_value  # new weights
            elif self._comission_fee_model == "trf":
                last_mu = 1
                mu = 1 - 2 * self._comission_fee_pct + self._comission_fee_pct**2
                while abs(mu - last_mu) > 1e-10:
                    last_mu = mu
                    mu = (
                        1
                        - self._comission_fee_pct * weights[0]
                        - (2 * self._comission_fee_pct - self._comission_fee_pct**2)
                        * np.sum(np.maximum(last_weights[1:] - mu * weights[1:], 0))
                    ) / (1 - self._comission_fee_pct * weights[0])
                self._portfolio_value = mu * self._portfolio_value

            # save initial portfolio value of this time step
            self._asset_memory["initial"].append(self._portfolio_value)

            # time passes and time variation changes the portfolio distribution
            portfolio = self._portfolio_value * (weights * self._price_variation)

            # calculate new portfolio value and weights
            self._portfolio_value = np.sum(portfolio)
            weights = portfolio / self._portfolio_value

            # save final portfolio value and weights of this time step
            self._asset_memory["final"].append(self._portfolio_value)
            self._final_weights.append(weights)

            # save date memory
            self._date_memory.append(self._info["end_time"])

            # define portfolio return
            rate_of_return = (
                self._asset_memory["final"][-1] / self._asset_memory["final"][-2]
            )
            portfolio_return = rate_of_return - 1
            portfolio_reward = np.log(rate_of_return)

            # save portfolio return memory
            self._portfolio_return_memory.append(portfolio_return)
            self._portfolio_reward_memory.append(portfolio_reward)

            # Define portfolio return
            self._reward = portfolio_reward
            self._reward = self._reward * self._reward_scaling

        if self._new_gym_api:
            return self._state, self._reward, self._terminal, False, self._info
        return self._state, self._reward, self._terminal, self._info

    def reset(self):
        """Resets the environment and returns it to its initial state (the
        fist date of the dataframe).

        Note:
            If the environment was created with "return_last_action" set to
            True, the initial state will be a Dict. If it's set to False,
            the initial state will be a Box. You can check the observation
            state through the attribute "observation_space".

        Returns:
            If "new_gym_api" is set to True, the following tuple is returned:
            (state, info). If it's set to False, only the initial state is
            returned.

            state: Initial state.
            info: Initial state info.
        """
        # time_index must start a little bit in the future to implement lookback
        self._time_index = self._time_window - 1
        self._reset_memory()

        self._state, self._info = self._get_state_and_info_from_time_index(
            self._time_index
        )
        self._portfolio_value = self._initial_amount
        self._terminal = False

        if self._new_gym_api:
            return self._state, self._info
        return self._state

    def _get_state_and_info_from_time_index(self, time_index):
        """Gets state and information given a time index. It also updates "data"
        attribute with information about the current simulation step.

        Args:
            time_index: An integer that represents the index of a specific datetime.
                The initial datetime of the dataframe is given by 0.

        Note:
            If the environment was created with "return_last_action" set to
            True, the returned state will be a Dict. If it's set to False,
            the returned state will be a Box. You can check the observation
            state through the attribute "observation_space".

        Returns:
            A tuple with the following form: (state, info).

            state: The state of the current time index. It can be a Box or a Dict.
            info: A dictionary with some informations about the current simulation
                step. The dict has the following keys::

                {
                "tics": List of ticker symbols,
                "start_time": Start time of current time window,
                "start_time_index": Index of start time of current time window,
                "end_time": End time of current time window,
                "end_time_index": Index of end time of current time window,
                "data": Data related to the current time window,
                "price_variation": Price variation of current time step
                }
        """
        # returns state in form (channels, tics, timesteps)
        end_time = self._sorted_times[time_index]
        start_time = self._sorted_times[time_index - (self._time_window - 1)]

        # define data to be used in this time step
        self._data = self._df[
            (self._df[self._time_column] >= start_time)
            & (self._df[self._time_column] <= end_time)
        ][[self._time_column, self._tic_column] + self._features]

        # define price variation of this time_step
        self._price_variation = self._df_price_variation[
            self._df_price_variation[self._time_column] == end_time
        ][self._valuation_feature].to_numpy()
        self._price_variation = np.insert(self._price_variation, 0, 1)

        # define state to be returned
        state = None
        for tic in self._tic_list:
            tic_data = self._data[self._data[self._tic_column] == tic]
            tic_data = tic_data[self._features].to_numpy().T
            tic_data = tic_data[..., np.newaxis]
            state = tic_data if state is None else np.append(state, tic_data, axis=2)
        state = state.transpose((0, 2, 1))
        info = {
            "tics": self._tic_list,
            "start_time": start_time,
            "start_time_index": time_index - (self._time_window - 1),
            "end_time": end_time,
            "end_time_index": time_index,
            "data": self._data,
            "price_variation": self._price_variation,
        }
        return self._standardize_state(state), info

    def render(self, mode="human"):
        """Renders the environment.

        Returns:
            Observation of current simulation step.
        """
        return self._state

    def _softmax_normalization(self, actions):
        """Normalizes the action vector using softmax function.

        Returns:
            Normalized action vector (portfolio vector).
        """
        numerator = np.exp(actions)
        denominator = np.sum(np.exp(actions))
        softmax_output = numerator / denominator
        return softmax_output

    def enumerate_portfolio(self):
        """Enumerates the current porfolio by showing the ticker symbols
        of all the investments considered in the portfolio.
        """
        print("Index: 0. Tic: Cash")
        for index, tic in enumerate(self._tic_list):
            print("Index: {}. Tic: {}".format(index + 1, tic))

    def _preprocess_data(self, order, normalize):
        """Orders and normalizes the environment's dataframe.

        Args:
            order: If true, the dataframe will be ordered by ticker list
                and datetime.
            normalize: Defines the normalization method applied to the dataframe.
                Possible values are "by_previous_time", "by_fist_time_window_value",
                "by_COLUMN_NAME" (where COLUMN_NAME must be changed to a real column
                name) and a custom function. If None no normalization is done.
        """
        # order time dataframe by tic and time
        if order:
            self._df = self._df.sort_values(by=[self._tic_column, self._time_column])
        # defining price variation after ordering dataframe
        self._df_price_variation = self._temporal_variation_df()
        # apply normalization
        if normalize:
            self._normalize_dataframe(normalize)
        # transform str to datetime
        self._df[self._time_column] = pd.to_datetime(self._df[self._time_column])
        self._df_price_variation[self._time_column] = pd.to_datetime(
            self._df_price_variation[self._time_column]
        )
        # transform numeric variables to float32 (compatibility with pytorch)
        self._df[self._features] = self._df[self._features].astype("float32")
        self._df_price_variation[self._features] = self._df_price_variation[
            self._features
        ].astype("float32")

    def _reset_memory(self):
        """Resets the environment's memory."""
        date_time = self._sorted_times[self._time_index]
        # memorize portfolio value each step
        self._asset_memory = {
            "initial": [self._initial_amount],
            "final": [self._initial_amount],
        }
        # memorize portfolio return and reward each step
        self._portfolio_return_memory = [0]
        self._portfolio_reward_memory = [0]
        # initial action: all money is allocated in cash
        self._actions_memory = [np.array([1] + [0] * self._stock_dim, dtype=np.float32)]
        # memorize portfolio weights at the ending of time step
        self._final_weights = [np.array([1] + [0] * self._stock_dim, dtype=np.float32)]
        # memorize datetimes
        self._date_memory = [date_time]

    def _standardize_state(self, state):
        """Standardize the state given the observation space. If "return_last_action"
        is set to False, a three-dimensional box is returned. If it's set to True, a
        dictionary is returned. The dictionary follows the standard below::

            {
            "state": Three-dimensional box representing the current state,
            "last_action": One-dimensional box representing the last action
            }
        """
        last_action = self._actions_memory[-1]
        if self._return_last_action:
            return {"state": state, "last_action": last_action}
        else:
            return state

    def _normalize_dataframe(self, normalize):
        """ "Normalizes the environment's dataframe.

        Args:
            normalize: Defines the normalization method applied to the dataframe.
                Possible values are "by_previous_time", "by_fist_time_window_value",
                "by_COLUMN_NAME" (where COLUMN_NAME must be changed to a real column
                name) and a custom function. If None no normalization is done.

        Note:
            If a custom function is used in the normalization, it must have an
            argument representing the environment's dataframe.
        """
        if type(normalize) == str:
            if normalize == "by_fist_time_window_value":
                print(
                    "Normalizing {} by first time window value...".format(
                        self._features
                    )
                )
                self._df = self._temporal_variation_df(self._time_window - 1)
            elif normalize == "by_previous_time":
                print("Normalizing {} by previous time...".format(self._features))
                self._df = self._temporal_variation_df()
            elif normalize.startswith("by_"):
                normalizer_column = normalize[3:]
                print("Normalizing {} by {}".format(self._features, normalizer_column))
                for column in self._features:
                    self._df[column] = self._df[column] / self._df[normalizer_column]
        elif callable(normalize):
            print("Applying custom normalization function...")
            self._df = normalize(self._df)
        else:
            print("No normalization was performed.")

    def _temporal_variation_df(self, periods=1):
        """Calculates the temporal variation dataframe. For each feature, this
        dataframe contains the rate of the current feature's value and the last
        feature's value given a period. It's used to normalize the dataframe.

        Args:
            periods: Periods (in time indexes) to calculate temporal variation.

        Returns:
            Temporal variation dataframe.
        """
        df_temporal_variation = self._df.copy()
        prev_columns = []
        for column in self._features:
            prev_column = "prev_{}".format(column)
            prev_columns.append(prev_column)
            df_temporal_variation[prev_column] = df_temporal_variation.groupby(
                self._tic_column
            )[column].shift(periods=periods)
            df_temporal_variation[column] = (
                df_temporal_variation[column] / df_temporal_variation[prev_column]
            )
        df_temporal_variation = (
            df_temporal_variation.drop(columns=prev_columns)
            .fillna(1)
            .reset_index(drop=True)
        )
        return df_temporal_variation

    def _seed(self, seed=None):
        """Seeds the sources of randomness of this environment to guarantee
        reproducibility.

        Args:
            seed: Seed value to be applied.

        Returns:
            Seed value applied.
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self, env_number=1):
        """Generates an environment compatible with Stable Baselines 3. The
        generated environment is a vectorized version of the current one.

        Returns:
            A tuple with the generated environment and an initial observation.
        """
        e = DummyVecEnv([lambda: self] * env_number)
        obs = e.reset()
        return e, obs
