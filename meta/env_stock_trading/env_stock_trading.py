import gym
import numpy as np
from numpy import random as rd


class StockTradingEnv(gym.Env):
    """
    多股票独立择时交易环境 - Multi-Stock Independent Timing Trading Environment
    
    交易模式 (Trading Mode):
    - 每个股票可以独立决策买入/卖出数量 - Each stock can independently decide buy/sell quantities
    - 支持多个股票同时交易 - Supports simultaneous trading of multiple stocks
    - 基于技术指标和价格信息进行择时 - Market timing based on technical indicators and price information
    
    应用场景 (Application Scenarios):
    - 股票投资组合的主动管理 - Active management of stock portfolios
    - 多因子选股与择时策略 - Multi-factor stock selection and timing strategies
    - 量化交易策略开发 - Quantitative trading strategy development
    
    核心特点 (Key Features):
    - 动作空间: 连续值，表示每只股票的买卖数量 - Action space: continuous values representing buy/sell quantities for each stock
    - 状态空间: 包含现金、股票价格、持仓量、技术指标 - State space: includes cash, stock prices, holdings, technical indicators
    - 奖励函数: 基于资产变化的收益率 - Reward function: based on asset change returns
    """
    def __init__(
        self,
        config,
        initial_account=1e6,  # 初始账户资金 - Initial account funds
        gamma=0.99,  # 奖励折扣因子 - Reward discount factor
        turbulence_thresh=99,  # 湍流阈值，用于风险控制 - Turbulence threshold for risk control
        min_stock_rate=0.1,  # 最小交易比例 - Minimum trading ratio
        max_stock=1e2,  # 最大股票交易数量 - Maximum stock trading quantity
        initial_capital=1e6,  # 初始资本 - Initial capital
        buy_cost_pct=1e-3,  # 买入手续费率 - Buy transaction fee rate
        sell_cost_pct=1e-3,  # 卖出手续费率 - Sell transaction fee rate
        reward_scaling=2**-11,  # 奖励缩放因子 - Reward scaling factor
        initial_stocks=None,  # 初始股票持仓 - Initial stock holdings
    ):
        price_array = config["price_array"]  # 价格数组 - Price array
        tech_array = config["tech_array"]    # 技术指标数组 - Technical indicator array
        turbulence_array = config["turbulence_array"]  # 湍流度数组 - Turbulence array
        if_train = config["if_train"]  # 是否为训练模式 - Whether in training mode
        self.price_array = price_array.astype(np.float32)
        self.tech_array = tech_array.astype(np.float32)
        self.turbulence_array = turbulence_array

        self.tech_array = self.tech_array * 2**-7  # 技术指标归一化 - Technical indicator normalization
        self.turbulence_bool = (turbulence_array > turbulence_thresh).astype(np.float32)  # 湍流布尔标记 - Turbulence boolean flag
        self.turbulence_array = (
            self.sigmoid_sign(turbulence_array, turbulence_thresh) * 2**-5
        ).astype(np.float32)

        stock_dim = self.price_array.shape[1]  # 股票维度 - Stock dimension
        self.gamma = gamma
        self.max_stock = max_stock
        self.min_stock_rate = min_stock_rate
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.initial_capital = initial_capital
        self.initial_stocks = (
            np.zeros(stock_dim, dtype=np.float32)
            if initial_stocks is None
            else initial_stocks
        )

        # reset()
        self.time = None  # 当前时间步 - Current time step
        self.cash = None  # 现金余额 - Cash balance
        self.stocks = None  # 股票持仓 - Stock holdings
        self.total_asset = None  # 总资产 - Total assets
        self.gamma_reward = None  # 折扣奖励 - Discounted reward
        self.initial_total_asset = None  # 初始总资产 - Initial total assets

        # environment information
        self.env_name = "StockEnv"
        # self.state_dim = 1 + 2 + 2 * stock_dim + self.tech_ary.shape[1]
        # # cash + (turbulence, turbulence_bool) + (price, stock) * stock_dim + tech_dim
        self.state_dim = 1 + 2 + 3 * stock_dim + self.tech_array.shape[1]
        # cash + (turbulence, turbulence_bool) + (price, stock) * stock_dim + tech_dim
        # 现金 + (湍流度, 湍流布尔值) + (价格, 股票持仓, 冷却时间) * 股票数量 + 技术指标维度
        self.stocks_cd = None  # 股票冷却时间 - Stock cooldown time
        self.action_dim = stock_dim  # 动作维度等于股票数量 - Action dimension equals number of stocks
        self.max_step = self.price_array.shape[0] - 1  # 最大步数 - Maximum steps
        self.if_train = if_train
        self.if_discrete = False  # 连续动作空间 - Continuous action space
        self.target_return = 10.0  # 目标收益率 - Target return
        self.episode_return = 0.0  # 当前轮次收益率 - Current episode return

        self.observation_space = gym.spaces.Box(
            low=-3000, high=3000, shape=(self.state_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(self.action_dim,), dtype=np.float32
        )

    def reset(self):
        """重置环境状态 - Reset environment state"""
        self.time = 0
        price = self.price_array[self.time]

        if self.if_train:
            # 训练模式下添加随机性 - Add randomness in training mode
            self.stocks = (
                self.initial_stocks + rd.randint(0, 64, size=self.initial_stocks.shape)
            ).astype(np.float32)
            self.stocks_cool_down = np.zeros_like(self.stocks)
            self.cash = (
                self.initial_capital * rd.uniform(0.95, 1.05)
                - (self.stocks * price).sum()
            )
        else:
            # 测试模式下使用固定初始值 - Use fixed initial values in test mode
            self.stocks = self.initial_stocks.astype(np.float32)
            self.stocks_cool_down = np.zeros_like(self.stocks)
            self.cash = self.initial_capital

        self.total_asset = self.cash + (self.stocks * price).sum()
        self.initial_total_asset = self.total_asset
        self.gamma_reward = 0.0
        return self.get_state(price)  # state

    def step(self, actions):
        """执行一步交易动作 - Execute one trading action step"""
        actions = (actions * self.max_stock).astype(int)  # 将动作缩放为整数股数 - Scale actions to integer shares
        self.time += 1
        price = self.price_array[self.time]
        self.stocks_cool_down += 1  # 增加冷却时间 - Increase cooldown time

        if self.turbulence_bool[self.time] == 0:  # 正常市场条件 - Normal market conditions
            min_action = int(self.max_stock * self.min_stock_rate)  # stock_cd
            for index in np.where(actions < -min_action)[0]:  # sell_index:
                if price[index] > 0:  # Sell only if current asset is > 0 # 只有在当前持仓大于0时才卖出
                    sell_num_shares = min(self.stocks[index], -actions[index])
                    self.stocks[index] -= sell_num_shares
                    self.cash += (
                        price[index] * sell_num_shares * (1 - self.sell_cost_pct)
                    )
                    self.stocks_cool_down[index] = 0
            for index in np.where(actions > min_action)[0]:  # buy_index:
                if (
                    price[index] > 0
                ):  # Buy only if the price is > 0 (no missing data in this particular date) # 只有在价格大于0时才买入(避免缺失数据)
                    buy_num_shares = min(self.cash // price[index], actions[index])
                    self.stocks[index] += buy_num_shares
                    self.cash -= price[index] * buy_num_shares * (1 + self.buy_cost_pct)
                    self.stocks_cool_down[index] = 0

        else:  # sell all when turbulence # 湍流期间卖出所有股票
            self.cash += (self.stocks * price).sum() * (1 - self.sell_cost_pct)
            self.stocks[:] = 0
            self.stocks_cool_down[:] = 0

        state = self.get_state(price)
        total_asset = self.cash + (self.stocks * price).sum()
        reward = (total_asset - self.total_asset) * self.reward_scaling  # 基于资产变化计算奖励 - Calculate reward based on asset change
        self.total_asset = total_asset

        self.gamma_reward = self.gamma_reward * self.gamma + reward  # 累积折扣奖励 - Accumulate discounted reward
        done = self.time == self.max_step
        if done:
            reward = self.gamma_reward
            self.episode_return = total_asset / self.initial_total_asset

        return state, reward, done, dict()

    def get_state(self, price):
        """获取当前状态 - Get current state"""
        cash = np.array(self.cash * (2**-12), dtype=np.float32)  # 现金归一化 - Cash normalization
        scale = np.array(2**-6, dtype=np.float32)  # 缩放因子 - Scale factor
        return np.hstack(
            (
                cash,  # 现金 - Cash
                self.turbulence_array[self.time],  # 湍流度 - Turbulence
                self.turbulence_bool[self.time],   # 湍流布尔值 - Turbulence boolean
                price * scale,                     # 归一化价格 - Normalized price
                self.stocks * scale,               # 归一化持仓 - Normalized holdings
                self.stocks_cool_down,             # 冷却时间 - Cooldown time
                self.tech_array[self.time],        # 技术指标 - Technical indicators
            )
        )  # state.astype(np.float32)

    @staticmethod
    def sigmoid_sign(ary, thresh):
        """Sigmoid符号函数用于湍流度处理 - Sigmoid sign function for turbulence processing"""
        def sigmoid(x):
            return 1 / (1 + np.exp(-x * np.e)) - 0.5

        return sigmoid(ary / thresh) * thresh
