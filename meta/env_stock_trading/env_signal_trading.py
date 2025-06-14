import gym
import matplotlib
import numpy as np
import pandas as pd
from gym import spaces
from gym.utils import seeding

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv


class SignalTradingEnv(gym.Env):
    """A single stock signal trading environment for OpenAI gym with discrete actions
    
    5分钟频率版本：专门为高频数据优化的信号交易环境，简化版本
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df,
        initial_amount,
        buy_cost_pct,
        sell_cost_pct,
        tech_indicator_list,
        turbulence_threshold=None,
        make_plots=False,
        print_verbosity=2,
        step=0,  # 改名为step，更适合5分钟频率
        initial=True,
        previous_state=[],
        model_name="",
        mode="",
        iteration="",
        random_seed=None,
        reward_config=None,
        frequency="5min",  # 新增：数据频率参数
    ):
        # 基本参数
        self.step_idx = step  # 当前步骤索引
        self.df = df
        self.initial_amount = initial_amount
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.tech_indicator_list = tech_indicator_list
        self.turbulence_threshold = turbulence_threshold
        self.make_plots = make_plots
        self.print_verbosity = print_verbosity
        self.initial = initial
        self.previous_state = previous_state
        self.model_name = model_name
        self.mode = mode
        self.iteration = iteration
        self.frequency = frequency

        # 5分钟频率相关参数
        self.freq_params = self._get_frequency_params(frequency)
        
        # 简化的奖励配置
        self.reward_config = reward_config or {
            'method': 'simple',  # 简化方法
            'return_weight': 1.0,
        }

        # 动作空间: 0=无持仓(卖出), 1=多头持有(买入)
        self.action_space = spaces.Discrete(2)
        
        # 状态空间计算 - 保持16维
        # [现金, 股价, 持仓状态(0/1), 股票数量, 技术指标..., 简化风控指标...]
        self.state_dim = 4 + len(self.tech_indicator_list) + 4  # +4 for simplified risk indicators
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,)
        )

        # 初始化数据
        self.data = self.df.loc[self.step_idx, :]
        self.terminal = False

        # 初始化交易变量
        self.immediate_reward = 0.0
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.episode = 0
        
        # 持仓状态: 0=无持仓, 1=多头持有
        self.position = 0
        
        # 记录变量
        self.portfolio_memory = []
        self.actions_memory = []
        self.date_memory = [self._get_date()]
        self.asset_memory = []
        
        # 简化的风控指标
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0
        self.peak_value = self.initial_amount
        self.period_returns = []
        
        # 简化的风控指标（用于状态空间）
        self.simple_risk_indicators = [0.0, 0.0, 0.0, 0.0]
        
        # 交易记录
        self.trade_start_value = None
        self.trade_start_step = None
        
        # 高频交易相关
        self.consecutive_holds = 0
        self.last_trade_step = 0

        # 初始化状态
        self.state = self._initiate_state()

    def _get_frequency_params(self, frequency):
        """根据数据频率设置相关参数"""
        if frequency == "5min":
            return {
                'periods_per_day': 78,  # 9:30-15:00，去除午休，每天78个5分钟
                'periods_per_year': 78 * 252,  # 年化基数
                'min_hold_periods': 6,         # 最小持仓周期（30分钟）
                'max_trade_freq': 0.1,         # 最大交易频率
            }
        elif frequency == "1min":
            return {
                'periods_per_day': 240,
                'periods_per_year': 240 * 252,
                'min_hold_periods': 30,
                'max_trade_freq': 0.05,
            }
        else:  # 默认日频
            return {
                'periods_per_day': 1,
                'periods_per_year': 252,
                'min_hold_periods': 1,
                'max_trade_freq': 1.0,
            }

    def _get_current_price(self):
        """获取当前股价"""
        return self.data.close

    def _get_total_asset(self):
        """获取当前总资产"""
        return self.state[0] + self.state[3] * self.state[1]

    def _calculate_simple_risk_indicators(self):
        """计算简化的风控指标 - 返回常数"""
        # 简化版本，只返回常数
        return 0.0, 0.0, 0.0, 0.0  # max_drawdown, current_drawdown, volatility, sharpe_ratio

    def _calculate_simple_reward(self, prev_value, current_value):
        """简化的奖励计算"""
        if prev_value > 0:
            return_rate = (current_value - prev_value) / prev_value
        else:
            return_rate = 0.0
            
        self.period_returns.append(return_rate)
        
        # 简化奖励：只基于收益率，放大适应5分钟频率
        reward = self.reward_config['return_weight'] * return_rate * 100
        
        # 基本交易成本惩罚
        if hasattr(self, 'prev_position') and self.prev_position != self.position:
            reward -= 0.0001  # 简单的交易成本
        
        return reward

    def step(self, action):
        self.terminal = self.step_idx >= len(self.df.index.unique()) - 1
        
        if self.terminal:
            if self.print_verbosity > 0:
                print(f"Episode: {self.episode}")
            if self.make_plots:
                self._make_plot()

            # 计算最终统计
            begin_total_asset = self.initial_amount
            end_total_asset = self._get_total_asset()
            total_pnl = end_total_asset - begin_total_asset
            total_return = total_pnl / begin_total_asset

            # 简化的最终风控指标
            sharpe_final = 0.0
            volatility_final = 0.0
            if len(self.period_returns) > 1:
                annual_factor = self.freq_params['periods_per_year']
                mean_return = np.mean(self.period_returns)
                std_return = np.std(self.period_returns)
                if std_return > 0:
                    sharpe_final = mean_return / std_return * np.sqrt(annual_factor)
                volatility_final = std_return * np.sqrt(annual_factor)

            # 交易频率统计
            total_periods = len(self.period_returns)
            trade_frequency = self.trades / total_periods if total_periods > 0 else 0

            if self.print_verbosity > 0 and self.episode % self.print_verbosity == 0:
                print(f"=== Episode {self.episode} Results (5-min frequency) ===")
                print(f"Total Return: {total_return:.4f}")
                print(f"Total P&L: {total_pnl:.2f}")
                print(f"Max Drawdown: {self.max_drawdown:.4f}")
                print(f"Sharpe Ratio: {sharpe_final:.4f}")
                print(f"Volatility: {volatility_final:.4f}")
                print(f"Total Trades: {self.trades}")
                print(f"Trade Frequency: {trade_frequency:.4f}")
                print("=" * 50)

            return self.state, self.immediate_reward, self.terminal, {
                'total_return': total_return,
                'total_pnl': total_pnl,
                'max_drawdown': self.max_drawdown,
                'sharpe': sharpe_final,
                'volatility': volatility_final,
                'total_trades': self.trades,
                'trade_frequency': trade_frequency,
                'data_frequency': self.frequency,
            }

        # 记录前一状态
        prev_value = self._get_total_asset()
        self.prev_position = self.position

        # 执行动作
        if action == 1:  # 买入/持有
            self._go_long()
        else:  # 卖出/保持无持仓
            self._go_neutral()

        # 计算即时奖励
        current_value = self._get_total_asset()
        self.immediate_reward = self._calculate_simple_reward(prev_value, current_value)

        # 更新简化风控指标
        self._update_simple_risk_indicators()

        # 更新turbulence
        if self.turbulence_threshold is not None:
            self.turbulence = self.data.get("turbulence", 0)

        # 记录 - 包含时间戳信息
        self.actions_memory.append(action)
        date = self._get_date()
        
        self.portfolio_memory.append({
            "date": date,
            "total_asset": current_value,
            "cash": self.state[0],
            "position": self.position,
            "stock_num": self.state[3],
            "price": self.state[1],
            "immediate_reward": self.immediate_reward,
            "action": action,
            "step_idx": self.step_idx,  # 5分钟频率特有
        })
        self.date_memory.append(date)

        # 更新下一个状态
        self.step_idx += 1
        self.data = self.df.loc[self.step_idx, :]
        self.state = self._update_state()

        return self.state, self.immediate_reward, self.terminal, {}

    def _update_simple_risk_indicators(self):
        """更新简化风控指标"""
        # 简化版本，只计算基本的回撤
        current_value = self._get_total_asset()
        if current_value > self.peak_value:
            self.peak_value = current_value
            self.current_drawdown = 0.0
        else:
            self.current_drawdown = (self.peak_value - current_value) / self.peak_value
        self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
        
        # 简化的风控指标
        self.simple_risk_indicators = [
            self.max_drawdown, 
            self.current_drawdown, 
            0.0,  # 简化的波动率
            0.0   # 简化的夏普比率
        ]

    def _go_long(self):
        """执行多头操作：满仓买入"""
        current_price = self._get_current_price()
        
        if current_price > 0 and self.position == 0:
            available_cash = self.state[0]
            cost_rate = 1 + self.buy_cost_pct
            max_shares = int(available_cash / (current_price * cost_rate))
            
            if max_shares > 0:
                buy_amount = max_shares * current_price
                cost = buy_amount * self.buy_cost_pct
                
                self.state[0] -= (buy_amount + cost)
                self.state[3] = max_shares
                self.position = 1
                
                self.cost += cost
                self.trades += 1

    def _go_neutral(self):
        """执行平仓操作：卖出所有股票"""
        current_price = self._get_current_price()
        
        if current_price > 0 and self.position == 1 and self.state[3] > 0:
            sell_amount = self.state[3] * current_price
            cost = sell_amount * self.sell_cost_pct
            
            self.state[0] += (sell_amount - cost)
            self.state[3] = 0
            self.position = 0
            
            self.cost += cost
            self.trades += 1

    def reset(self):
        # 重置所有变量
        self.step_idx = 0
        self.data = self.df.loc[self.step_idx, :]
        self.state = self._initiate_state()
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.terminal = False
        self.position = 0
        
        # 重置记录
        self.actions_memory = []
        self.date_memory = [self._get_date()]
        self.portfolio_memory = []
        self.asset_memory = []
        
        # 重置风控指标
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0
        self.peak_value = self.initial_amount
        self.period_returns = []
        self.trade_start_value = None
        self.trade_start_step = None
        
        # 重置简化指标
        self.simple_risk_indicators = [0.0, 0.0, 0.0, 0.0]
        
        # 重置高频交易相关
        self.consecutive_holds = 0
        self.last_trade_step = 0

        self.episode += 1
        return self.state

    def _initiate_state(self):
        """初始化状态 - 16维"""
        if self.initial:
            tech_indicators = [self.data[tech] for tech in self.tech_indicator_list]
            
            state = [self.initial_amount, self.data.close, 0, 0] + tech_indicators + self.simple_risk_indicators
        else:
            tech_indicators = [self.data[tech] for tech in self.tech_indicator_list]
            
            state = [
                self.previous_state[0],
                self.data.close,
                self.previous_state[2],
                self.previous_state[3],
            ] + tech_indicators + self.simple_risk_indicators
            
        return state

    def _update_state(self):
        """更新状态 - 16维"""
        tech_indicators = [self.data[tech] for tech in self.tech_indicator_list]
        
        state = [
            self.state[0],
            self.data.close,
            self.position,
            self.state[3],
        ] + tech_indicators + self.simple_risk_indicators
        
        return state

    def _get_date(self):
        """获取当前日期时间"""
        if hasattr(self.data, 'time'):
            return self.data.time
        elif hasattr(self.data, 'datetime'):
            return self.data.datetime
        else:
            return self.df.index[self.step_idx]

    def render(self, mode="human", close=False):
        return self.state

    def get_portfolio_df(self):
        """获取投资组合DataFrame"""
        portfolio_df = pd.DataFrame(self.portfolio_memory)
        if not portfolio_df.empty:
            portfolio_df["date"] = pd.to_datetime(portfolio_df["date"])
            portfolio_df.sort_values("date", inplace=True)
        return portfolio_df

    def save_asset_memory(self):
        """保存资产记录"""
        portfolio_df = self.get_portfolio_df()
        if not portfolio_df.empty:
            df_account_value = portfolio_df[["date", "total_asset"]].rename(
                columns={"total_asset": "account_value"}
            )
            return df_account_value
        else:
            return pd.DataFrame(columns=["date", "account_value"])

    def save_action_memory(self):
        """保存动作记录 - 5分钟频率"""
        if len(self.date_memory) > 1 and len(self.actions_memory) > 0:
            # 确保日期和动作数量匹配
            min_length = min(len(self.date_memory) - 1, len(self.actions_memory))
            date_list = self.date_memory[:min_length]
            action_list = self.actions_memory[:min_length]
            
            df_actions = pd.DataFrame({
                "date": date_list, 
                "actions": action_list
            })
            return df_actions
        else:
            return pd.DataFrame(columns=["date", "actions"])

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs

    def _make_plot(self):
        """生成图表 - 5分钟频率优化"""
        portfolio_df = self.get_portfolio_df()
        if not portfolio_df.empty:
            plt.figure(figsize=(15, 10))
            
            # 资产曲线
            plt.subplot(2, 2, 1)
            plt.plot(portfolio_df["date"], portfolio_df["total_asset"], color="r", label="Portfolio Value", linewidth=0.8)
            plt.title("Portfolio Value Over Time (5-min)")
            plt.xlabel("Time")
            plt.ylabel("Asset Value")
            plt.legend()
            plt.xticks(rotation=45)
            
            # 持仓状态
            plt.subplot(2, 2, 2)
            plt.plot(portfolio_df["date"], portfolio_df["position"], color="b", label="Position", linewidth=0.8)
            plt.title("Position Over Time")
            plt.xlabel("Time")
            plt.ylabel("Position (0=Neutral, 1=Long)")
            plt.legend()
            plt.xticks(rotation=45)
            
            # 动作分布
            plt.subplot(2, 2, 3)
            action_counts = portfolio_df["action"].value_counts()
            plt.bar(action_counts.index, action_counts.values)
            plt.title("Action Distribution")
            plt.xlabel("Action (0=Sell/Hold, 1=Buy/Hold)")
            plt.ylabel("Count")
            
            # 累积收益率
            plt.subplot(2, 2, 4)
            portfolio_df['strategy_cumret'] = (portfolio_df['total_asset'] / self.initial_amount - 1) * 100
            plt.plot(portfolio_df["date"], portfolio_df['strategy_cumret'], color="r", label="Strategy", linewidth=1)
            plt.title("Cumulative Return (%)")
            plt.xlabel("Time")
            plt.ylabel("Cumulative Return (%)")
            plt.legend()
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.savefig(f"results/signal_trading_5min_analysis_{self.episode}.png", dpi=150, bbox_inches='tight')
            plt.close() 