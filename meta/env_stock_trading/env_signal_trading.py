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
    """A single stock signal trading environment for OpenAI gym with discrete actions"""

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
        day=0,
        initial=True,
        previous_state=[],
        model_name="",
        mode="",
        iteration="",
        random_seed=None,
        reward_weights=None,  # 奖励权重配置
    ):
        # 基本参数
        self.day = day
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

        # 奖励函数权重
        self.reward_weights = reward_weights or {
            'return': 1.0,
            'sharpe': 0.1,
            'max_drawdown': -0.5,
            'volatility': -0.1
        }

        # 动作空间: 0=无持仓(卖出), 1=多头持有(买入)
        self.action_space = spaces.Discrete(2)
        
        # 状态空间计算
        # [现金, 股价, 持仓状态(0/1), 股票数量, 技术指标...，风控指标...]
        self.state_dim = 4 + len(self.tech_indicator_list) + 4  # +4 for risk indicators
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,)
        )

        # 初始化数据
        self.data = self.df.loc[self.day, :]
        self.terminal = False

        # 初始化状态
        self.state = self._initiate_state()

        # 初始化交易变量
        self.reward = 0
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
        
        # 风控指标
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0
        self.peak_value = self.initial_amount
        self.trade_returns = []  # 每笔交易的收益率
        self.daily_returns = []
        
        # 交易记录
        self.trade_start_value = None
        self.trade_start_day = None

    def _get_current_price(self):
        """获取当前股价"""
        return self.data.close

    def _get_total_asset(self):
        """获取当前总资产"""
        return self.state[0] + self.state[3] * self.state[1]  # 现金 + 股票数量 * 股价

    def _calculate_risk_indicators(self):
        """计算风控指标"""
        current_value = self._get_total_asset()
        
        # 更新峰值
        if current_value > self.peak_value:
            self.peak_value = current_value
            self.current_drawdown = 0.0
        else:
            self.current_drawdown = (self.peak_value - current_value) / self.peak_value
            
        # 更新最大回撤
        self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
        
        # 计算波动率（基于最近20天的日收益率）
        if len(self.daily_returns) > 1:
            recent_returns = self.daily_returns[-20:] if len(self.daily_returns) >= 20 else self.daily_returns
            volatility = np.std(recent_returns) * np.sqrt(252)  # 年化波动率
        else:
            volatility = 0.0
            
        # 计算夏普比率（基于最近收益率）
        if len(self.daily_returns) > 1 and np.std(self.daily_returns) > 0:
            sharpe = np.mean(self.daily_returns) / np.std(self.daily_returns) * np.sqrt(252)
        else:
            sharpe = 0.0
            
        return self.max_drawdown, self.current_drawdown, volatility, sharpe

    def _calculate_reward(self, prev_value, current_value):
        """计算综合奖励, 有没有shift 1 问题?"""
        # 基础收益率
        if prev_value > 0:
            return_rate = (current_value - prev_value) / prev_value
        else:
            return_rate = 0.0
            
        self.daily_returns.append(return_rate)
        
        # 获取风控指标
        max_dd, current_dd, volatility, sharpe = self._calculate_risk_indicators()
        
        # 计算综合奖励
        reward = (
            self.reward_weights['return'] * return_rate +
            self.reward_weights['sharpe'] * sharpe * 0.01 +  # 缩放sharpe
            self.reward_weights['max_drawdown'] * max_dd +
            self.reward_weights['volatility'] * volatility * 0.01  # 缩放volatility
        )
        
        # 对持仓变化给予额外奖励/惩罚
        if hasattr(self, 'prev_position'):
            if self.prev_position != self.position:
                # 交易成本惩罚
                reward -= 0.001
                
                # 如果是从持仓到无持仓，记录这笔交易的收益
                if self.prev_position == 1 and self.position == 0:
                    if self.trade_start_value is not None:
                        trade_return = (current_value - self.trade_start_value) / self.trade_start_value
                        self.trade_returns.append(trade_return)
                        
                        # 对profitable trades给额外奖励
                        if trade_return > 0:
                            reward += 0.01 * trade_return
                        else:
                            reward += 0.005 * trade_return  # 减少损失的惩罚
                            
                # 如果是从无持仓到持仓，记录交易开始
                elif self.prev_position == 0 and self.position == 1:
                    self.trade_start_value = current_value
                    self.trade_start_day = self.day
        
        return reward

    def step(self, action):
        self.terminal = self.day >= len(self.df.index.unique()) - 1
        
        if self.terminal:
            print(f"Episode: {self.episode}")
            if self.make_plots:
                self._make_plot()

            # 计算最终统计
            portfolio_df = self.get_portfolio_df()
            begin_total_asset = self.initial_amount
            end_total_asset = self._get_total_asset()
            tot_reward = end_total_asset - begin_total_asset
            total_return = tot_reward / begin_total_asset

            # 计算最终风控指标
            if len(self.daily_returns) > 1:
                sharpe_final = np.mean(self.daily_returns) / np.std(self.daily_returns) * np.sqrt(252)
                volatility_final = np.std(self.daily_returns) * np.sqrt(252)
            else:
                sharpe_final = 0.0
                volatility_final = 0.0

            if self.print_verbosity > 0 and self.episode % self.print_verbosity == 0:
                print(f"=== Episode {self.episode} Results ===")
                print(f"Total Return: {total_return:.4f}")
                print(f"Total Reward: {tot_reward:.2f}")
                print(f"Max Drawdown: {self.max_drawdown:.4f}")
                print(f"Sharpe Ratio: {sharpe_final:.4f}")
                print(f"Volatility: {volatility_final:.4f}")
                print(f"Total Trades: {self.trades}")
                print(f"Profitable Trades: {sum(1 for r in self.trade_returns if r > 0)}/{len(self.trade_returns)}")
                print("=================================")

            return self.state, self.reward, self.terminal, {
                'total_return': total_return,
                'max_drawdown': self.max_drawdown,
                'sharpe': sharpe_final,
                'volatility': volatility_final,
                'total_trades': self.trades
            }

        # 记录前一状态
        prev_value = self._get_total_asset()
        self.prev_position = self.position

        # 执行动作
        if action == 1:  # 买入/持有
            self._go_long()
        else:  # 卖出/保持无持仓
            self._go_neutral()

        # 计算奖励
        current_value = self._get_total_asset()
        self.reward = self._calculate_reward(prev_value, current_value)

        # 更新turbulence
        if self.turbulence_threshold is not None:
            self.turbulence = self.data.get("turbulence", 0)

        # 记录
        self.actions_memory.append(action)
        date = self._get_date()
        
        self.portfolio_memory.append({
            "date": date,
            "total_asset": current_value,
            "cash": self.state[0],
            "position": self.position,
            "stock_num": self.state[3],
            "price": self.state[1],
            "reward": self.reward,
            "action": action,
        })
        self.date_memory.append(date)

        # 更新下一个状态
        self.day += 1
        self.data = self.df.loc[self.day, :]
        self.state = self._update_state()

        return self.state, self.reward, self.terminal, {}

    def _go_long(self):
        """执行多头操作：满仓买入"""
        current_price = self._get_current_price()
        
        if current_price > 0 and self.position == 0:  # 只有在无持仓时才买入
            # 计算可买入股数
            available_cash = self.state[0]
            cost_rate = 1 + self.buy_cost_pct
            max_shares = int(available_cash / (current_price * cost_rate))
            
            if max_shares > 0:
                # 执行买入
                buy_amount = max_shares * current_price
                cost = buy_amount * self.buy_cost_pct
                
                self.state[0] -= (buy_amount + cost)  # 减少现金
                self.state[3] = max_shares  # 设置股票数量
                self.position = 1  # 更新持仓状态
                
                self.cost += cost
                self.trades += 1

    def _go_neutral(self):
        """执行平仓操作：卖出所有股票"""
        current_price = self._get_current_price()
        
        if current_price > 0 and self.position == 1 and self.state[3] > 0:
            # 卖出所有股票
            sell_amount = self.state[3] * current_price
            cost = sell_amount * self.sell_cost_pct
            
            self.state[0] += (sell_amount - cost)  # 增加现金
            self.state[3] = 0  # 清空股票
            self.position = 0  # 更新持仓状态
            
            self.cost += cost
            self.trades += 1

    def reset(self):
        # 重置所有变量
        self.day = 0
        self.data = self.df.loc[self.day, :]
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
        self.trade_returns = []
        self.daily_returns = []
        self.trade_start_value = None
        self.trade_start_day = None

        self.episode += 1
        return self.state

    def _initiate_state(self):
        """初始化状态"""
        if self.initial:
            # 初始状态: [现金, 股价, 持仓状态, 股票数量, 技术指标..., 风控指标...]
            tech_indicators = [self.data[tech] for tech in self.tech_indicator_list]
            risk_indicators = [0.0, 0.0, 0.0, 0.0]  # [max_drawdown, current_drawdown, volatility, sharpe]
            
            state = [self.initial_amount, self.data.close, 0, 0] + tech_indicators + risk_indicators
        else:
            # 使用之前的状态
            tech_indicators = [self.data[tech] for tech in self.tech_indicator_list]
            risk_indicators = self._calculate_risk_indicators()
            
            state = [
                self.previous_state[0],  # 现金
                self.data.close,         # 当前股价
                self.previous_state[2],  # 持仓状态
                self.previous_state[3],  # 股票数量
            ] + tech_indicators + list(risk_indicators)
            
        return state

    def _update_state(self):
        """更新状态"""
        tech_indicators = [self.data[tech] for tech in self.tech_indicator_list]
        risk_indicators = self._calculate_risk_indicators()
        
        state = [
            self.state[0],      # 现金
            self.data.close,    # 当前股价
            self.position,      # 持仓状态
            self.state[3],      # 股票数量
        ] + tech_indicators + list(risk_indicators)
        
        return state

    def _get_date(self):
        """获取当前日期"""
        return self.data.time

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
        """保存动作记录"""
        if len(self.date_memory) > 1 and len(self.actions_memory) > 0:
            date_list = self.date_memory[:-1]
            action_list = self.actions_memory
            df_actions = pd.DataFrame({"date": date_list, "actions": action_list})
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
        """生成图表"""
        portfolio_df = self.get_portfolio_df()
        if not portfolio_df.empty:
            plt.figure(figsize=(12, 8))
            
            # 资产曲线
            plt.subplot(2, 2, 1)
            plt.plot(portfolio_df["date"], portfolio_df["total_asset"], color="r", label="Portfolio Value")
            plt.title("Portfolio Value Over Time")
            plt.xlabel("Date")
            plt.ylabel("Asset Value")
            plt.legend()
            
            # 持仓状态
            plt.subplot(2, 2, 2)
            plt.plot(portfolio_df["date"], portfolio_df["position"], color="b", label="Position")
            plt.title("Position Over Time")
            plt.xlabel("Date")
            plt.ylabel("Position (0=Neutral, 1=Long)")
            plt.legend()
            
            # 动作分布
            plt.subplot(2, 2, 3)
            action_counts = portfolio_df["action"].value_counts()
            plt.bar(action_counts.index, action_counts.values)
            plt.title("Action Distribution")
            plt.xlabel("Action (0=Sell/Hold, 1=Buy/Hold)")
            plt.ylabel("Count")
            
            # 收益分布
            plt.subplot(2, 2, 4)
            plt.plot(portfolio_df["date"], portfolio_df["reward"], color="g", label="Daily Reward")
            plt.title("Daily Rewards")
            plt.xlabel("Date")
            plt.ylabel("Reward")
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(f"results/signal_trading_analysis_{self.episode}.png", dpi=150)
            plt.close() 