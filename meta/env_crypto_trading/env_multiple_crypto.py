import math

import numpy as np


class CryptoEnv:  # custom env # 自定义环境
    """
    多加密货币交易环境 - Multi-Cryptocurrency Trading Environment
    
    交易模式 (Trading Mode):
    - 多种加密货币同时交易 - Trading multiple cryptocurrencies simultaneously
    - 基于价格和技术指标的连续动作交易 - Continuous action trading based on price and technical indicators
    - 支持买入卖出的连续数量决策 - Supports continuous quantity decisions for buy/sell
    - 考虑加密货币价格的巨大差异进行动作归一化 - Normalizes actions considering huge price differences in cryptocurrencies
    
    应用场景 (Application Scenarios):
    - 加密货币投资组合管理 - Cryptocurrency portfolio management
    - 数字资产的量化交易策略 - Quantitative trading strategies for digital assets
    - 加密货币市场的套利交易 - Arbitrage trading in cryptocurrency markets
    - DeFi和数字资产配置 - DeFi and digital asset allocation
    
    核心特点 (Key Features):
    - 动作空间: 连续值，表示各加密货币的买卖数量 - Action space: continuous values representing buy/sell quantities
    - 状态空间: 现金 + 持仓 + 历史价格和技术指标 - State space: cash + holdings + historical prices and indicators
    - 奖励函数: 基于总资产变化的gamma折扣奖励 - Reward function: gamma-discounted reward based on total asset changes
    - 特殊处理: 针对加密货币的价格数量级差异进行动作归一化 - Special handling: action normalization for price magnitude differences
    """
    def __init__(
        self,
        config,  # 配置字典 - Configuration dictionary
        lookback=1,  # 回看窗口大小 - Lookback window size
        initial_capital=1e6,  # 初始资本 - Initial capital
        buy_cost_pct=1e-3,  # 买入手续费率 - Buy cost percentage
        sell_cost_pct=1e-3,  # 卖出手续费率 - Sell cost percentage
        gamma=0.99,  # 奖励折扣因子 - Reward discount factor
    ):
        self.lookback = lookback  # 历史数据回看期 - Historical data lookback period
        self.initial_total_asset = initial_capital  # 初始总资产 - Initial total assets
        self.initial_cash = initial_capital  # 初始现金 - Initial cash
        self.buy_cost_pct = buy_cost_pct  # 买入成本百分比 - Buy cost percentage
        self.sell_cost_pct = sell_cost_pct  # 卖出成本百分比 - Sell cost percentage
        self.max_stock = 1  # 最大股票持仓 - Maximum stock holding
        self.gamma = gamma  # 折扣因子 - Discount factor
        self.price_array = config["price_array"]  # 价格数组 - Price array
        self.tech_array = config["tech_array"]    # 技术指标数组 - Technical indicator array
        self._generate_action_normalizer()  # 生成动作归一化器 - Generate action normalizer
        self.crypto_num = self.price_array.shape[1]  # 加密货币数量 - Number of cryptocurrencies
        self.max_step = self.price_array.shape[0] - lookback - 1  # 最大步数 - Maximum steps

        # reset # 重置变量
        self.time = lookback - 1  # 当前时间 - Current time
        self.cash = self.initial_cash  # 当前现金 - Current cash
        self.current_price = self.price_array[self.time]  # 当前价格 - Current price
        self.current_tech = self.tech_array[self.time]    # 当前技术指标 - Current technical indicators
        self.stocks = np.zeros(self.crypto_num, dtype=np.float32)  # 股票持仓数组 - Stock holdings array

        self.total_asset = self.cash + (self.stocks * self.price_array[self.time]).sum()  # 总资产 - Total assets
        self.episode_return = 0.0  # 回合收益率 - Episode return
        self.gamma_return = 0.0    # Gamma折扣收益 - Gamma discounted return

        """env information""" # 环境信息
        self.env_name = "MulticryptoEnv"  # 环境名称 - Environment name
        self.state_dim = (
            1 + (self.price_array.shape[1] + self.tech_array.shape[1]) * lookback
        )  # 状态维度：1(现金) + (价格维度 + 技术指标维度) * 回看期 - State dimension: 1(cash) + (price_dim + tech_dim) * lookback
        self.action_dim = self.price_array.shape[1]  # 动作维度等于加密货币数量 - Action dimension equals number of cryptocurrencies
        self.if_discrete = False  # 连续动作空间 - Continuous action space
        self.target_return = 10   # 目标收益率 - Target return

    def reset(self) -> np.ndarray:
        """重置环境到初始状态 - Reset environment to initial state"""
        self.time = self.lookback - 1  # 重置时间 - Reset time
        self.current_price = self.price_array[self.time]  # 重置当前价格 - Reset current price
        self.current_tech = self.tech_array[self.time]    # 重置当前技术指标 - Reset current technical indicators
        self.cash = self.initial_cash  # reset() # 重置现金
        self.stocks = np.zeros(self.crypto_num, dtype=np.float32)  # 重置持仓 - Reset holdings
        self.total_asset = self.cash + (self.stocks * self.price_array[self.time]).sum()  # 重置总资产 - Reset total assets

        return self.get_state()

    def step(self, actions) -> (np.ndarray, float, bool, None):
        """执行一步交易动作 - Execute one trading action step"""
        self.time += 1  # 时间步进 - Time step forward

        price = self.price_array[self.time]  # 获取当前价格 - Get current price
        for i in range(self.action_dim):
            norm_vector_i = self.action_norm_vector[i]  # 获取归一化向量 - Get normalization vector
            actions[i] = actions[i] * norm_vector_i  # 应用归一化 - Apply normalization

        for index in np.where(actions < 0)[0]:  # sell_index: # 卖出索引
            if price[index] > 0:  # Sell only if current asset is > 0 # 只有在当前资产大于0时才卖出
                sell_num_shares = min(self.stocks[index], -actions[index])  # 计算卖出数量 - Calculate sell quantity
                self.stocks[index] -= sell_num_shares  # 减少持仓 - Reduce holdings
                self.cash += price[index] * sell_num_shares * (1 - self.sell_cost_pct)  # 增加现金(扣除手续费) - Increase cash (minus fees)

        for index in np.where(actions > 0)[0]:  # buy_index: # 买入索引
            if (
                price[index] > 0
            ):  # Buy only if the price is > 0 (no missing data in this particular date) # 只有在价格大于0时才买入(该日期无缺失数据)
                buy_num_shares = min(self.cash // price[index], actions[index])  # 计算买入数量 - Calculate buy quantity
                self.stocks[index] += buy_num_shares  # 增加持仓 - Increase holdings
                self.cash -= price[index] * buy_num_shares * (1 + self.buy_cost_pct)  # 减少现金(加上手续费) - Decrease cash (plus fees)

        """update time""" # 更新时间
        done = self.time == self.max_step  # 检查是否结束 - Check if done
        state = self.get_state()  # 获取新状态 - Get new state
        next_total_asset = self.cash + (self.stocks * self.price_array[self.time]).sum()  # 计算新的总资产 - Calculate new total assets
        reward = (next_total_asset - self.total_asset) * 2**-16  # 计算奖励(归一化) - Calculate reward (normalized)
        self.total_asset = next_total_asset  # 更新总资产 - Update total assets
        self.gamma_return = self.gamma_return * self.gamma + reward  # 更新gamma折扣收益 - Update gamma discounted return
        self.cumu_return = self.total_asset / self.initial_cash  # 累积收益率 - Cumulative return
        if done:
            reward = self.gamma_return  # 结束时使用gamma收益 - Use gamma return when done
            self.episode_return = self.total_asset / self.initial_cash  # 计算回合收益率 - Calculate episode return
        return state, reward, done, None

    def get_state(self):
        """获取当前环境状态 - Get current environment state"""
        state = np.hstack((self.cash * 2**-18, self.stocks * 2**-3))  # 现金和持仓归一化 - Normalize cash and holdings
        for i in range(self.lookback):
            tech_i = self.tech_array[self.time - i]  # 获取历史技术指标 - Get historical technical indicators
            normalized_tech_i = tech_i * 2**-15  # 归一化技术指标 - Normalize technical indicators
            state = np.hstack((state, normalized_tech_i)).astype(np.float32)  # 合并状态 - Concatenate state
        return state

    def close(self):
        """关闭环境 - Close environment"""
        pass

    def _generate_action_normalizer(self):
        """生成动作归一化器以适应加密货币的巨大价格差异 - Generate action normalizer to adapt to huge price differences in cryptocurrencies"""
        # normalize action to adjust for large price differences in cryptocurrencies # 归一化动作以适应加密货币的巨大价格差异
        action_norm_vector = []
        price_0 = self.price_array[0]  # Use row 0 prices to normalize # 使用第0行价格进行归一化
        for price in price_0:
            x = math.floor(math.log(price, 10))  # the order of magnitude # 数量级
            action_norm_vector.append(1 / ((10) ** x))  # 计算归一化因子 - Calculate normalization factor

        action_norm_vector = (
            np.asarray(action_norm_vector) * 10000
        )  # roughly control the maximum transaction amount for each action # 大致控制每个动作的最大交易金额
        self.action_norm_vector = np.asarray(action_norm_vector)  # 保存归一化向量 - Save normalization vector
