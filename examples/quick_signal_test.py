"""
快速信号交易测试脚本
用于快速测试信号交易环境和策略
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import os
import pandas as pd
import numpy as np
import random
import torch
import pickle
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def set_global_seed(seed=42):
    """设置全局随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

set_global_seed(42)

try:
    from meta import config
    from meta.data_processor import DataProcessor
    from main import check_and_make_directories
    from meta.env_stock_trading.env_signal_trading import SignalTradingEnv
    from agents.stablebaselines3_models import DRLAgent
    from meta.config import (
        DATA_SAVE_DIR,
        TRAINED_MODEL_DIR,
        TENSORBOARD_LOG_DIR,
        RESULTS_DIR,
        INDICATORS,
    )
    print("✓ 所有模块导入成功!")
except ImportError as e:
    print(f"✗ 导入模块失败: {e}")
    print("请确保您在正确的项目目录中运行此脚本")
    exit(1)

# 策略配置（直接在文件中定义，避免导入问题）
REWARD_CONFIGS = {
    "information_ratio": {
        'method': 'information_ratio',
        'return_weight': 1.0,
        'risk_penalty_weight': 0.5,
        'trade_quality_weight': 0.1,
        'final_reward_weight': 2.0,
        'benchmark': 'buy_hold'
    },
    "multi_factor": {
        'method': 'multi_factor',
        'return_weight': 1.0,
        'risk_penalty_weight': 0.8,
        'trade_quality_weight': 0.15,
        'final_reward_weight': 1.5,
        'benchmark': 'buy_hold'
    }
}

PPO_CONFIG = {
    "learning_rate": 0.0003,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
}

def quick_data_preparation():
    """快速数据准备"""
    print("=" * 50)
    print("开始数据准备...")
    
    # 使用更短的时间段进行快速测试
    TRAIN_START = "2019-01-01"
    TRAIN_END = "2019-06-01"
    TRADE_START = "2019-06-01"
    TRADE_END = "2019-12-01"
    
    ticker_list = ["600000.SH"]  # 单股票测试
    
    # 检查是否有缓存数据
    cache_file = "datasets/quick_test_data.pkl"
    if os.path.exists(cache_file):
        print("发现缓存数据，正在加载...")
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            return cached_data['train'], cached_data['trade']
        except Exception as e:
            print(f"加载缓存数据失败: {e}")
            print("将重新创建数据...")
    
    # 创建必要目录
    check_and_make_directories([DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR])
    
    # 尝试下载真实数据
    try:
        kwargs = {"token": "d820068d786c287ae44926d3ead93673337ca6569cc9c81be44dbcbd"}
        p = DataProcessor(
            data_source="tushare",
            start_date=TRAIN_START,
            end_date=TRADE_END,
            time_interval="1d",
            **kwargs,
        )
        
        print(f"下载数据: {ticker_list[0]} ({TRAIN_START} 到 {TRADE_END})")
        p.download_data(ticker_list=ticker_list)
        p.clean_data()
        p.fillna()
        
        print("添加技术指标...")
        p.add_technical_indicator(config.INDICATORS)
        p.fillna()
        
        print(f"数据形状: {p.dataframe.shape}")
        
        # 数据分割
        train = p.data_split(p.dataframe, TRAIN_START, TRAIN_END)
        trade = p.data_split(p.dataframe, TRADE_START, TRADE_END)
        
        print(f"训练数据: {train.shape}, 交易数据: {trade.shape}")
        
        # 缓存数据
        os.makedirs("datasets", exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump({'train': train, 'trade': trade}, f)
        print(f"数据已缓存到: {cache_file}")
        
        return train, trade
        
    except Exception as e:
        print(f"数据准备失败: {e}")
        print("使用模拟数据进行测试...")
        return create_mock_data()

def create_mock_data():
    """创建模拟数据用于测试"""
    print("创建模拟数据...")
    
    # 创建基础时间序列
    dates = pd.date_range('2019-01-01', '2019-12-01', freq='D')
    n_days = len(dates)
    
    # 生成模拟价格数据（带趋势和噪声）
    np.random.seed(42)
    base_price = 10.0
    trend = np.linspace(0, 2, n_days)  # 轻微上升趋势
    noise = np.random.normal(0, 0.5, n_days)
    prices = base_price + trend + noise
    prices = np.maximum(prices, 1.0)  # 确保价格为正
    
    # 生成技术指标
    data = []
    for i, (date, price) in enumerate(zip(dates, prices)):
        # 基础数据
        row = {
            'time': date.strftime('%Y-%m-%d'),
            'tic': '600000.SH',
            'close': price,
            'high': price * (1 + np.random.uniform(0, 0.02)),
            'low': price * (1 - np.random.uniform(0, 0.02)),
            'open': price * (1 + np.random.uniform(-0.01, 0.01)),
            'volume': np.random.uniform(1000000, 5000000),
        }
        
        # 技术指标（简化版）
        lookback = min(i + 1, 20)
        recent_prices = prices[max(0, i-lookback+1):i+1]
        
        row.update({
            'macd': np.mean(recent_prices[-12:]) - np.mean(recent_prices) if len(recent_prices) >= 12 else 0,
            'boll_ub': np.mean(recent_prices) + 2 * np.std(recent_prices),
            'boll_lb': np.mean(recent_prices) - 2 * np.std(recent_prices),
            'rsi_30': 50 + np.random.uniform(-20, 20),  # 简化RSI
            'cci_30': np.random.uniform(-100, 100),
            'dx_30': np.random.uniform(10, 90),
            'close_30_sma': np.mean(recent_prices),
            'close_60_sma': np.mean(recent_prices),
        })
        
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # 分割数据
    split_date = '2019-06-01'
    train = df[df['time'] < split_date].copy()
    trade = df[df['time'] >= split_date].copy()
    
    print(f"模拟数据创建完成 - 训练: {len(train)}, 交易: {len(trade)}")
    return train, trade

def quick_train_test(train_data, trade_data):
    """快速训练和测试"""
    print("=" * 50)
    print(f"开始快速训练测试")
    
    try:
        # 环境配置
        env_kwargs = {
            "initial_amount": 100000,  # 降低初始金额加快测试
            "buy_cost_pct": 1.25e-3,
            "sell_cost_pct": 1.25e-3,
            "tech_indicator_list": config.INDICATORS,
            "turbulence_threshold": None,
            "make_plots": False,  # 快速测试时不绘图
            "print_verbosity": 1,
            "reward_config": REWARD_CONFIGS["information_ratio"],
            "random_seed": 42,
        }
        
        # 创建训练环境
        print("创建训练环境...")
        e_train_gym = SignalTradingEnv(df=train_data, **env_kwargs)
        env_train, _ = e_train_gym.get_sb_env()
        
        print(f"✓ 动作空间: {e_train_gym.action_space}")
        print(f"✓ 观测空间维度: {e_train_gym.observation_space.shape}")
        
        # 使用PPO进行快速训练
        print(f"训练模型: PPO")
        print(f"模型配置: {PPO_CONFIG}")
        
        # 创建代理并训练
        agent = DRLAgent(env=env_train)
        model = agent.get_model("ppo", model_kwargs=PPO_CONFIG, seed=42)
        
        # 快速训练（减少步数）
        quick_timesteps = 5000
        print(f"开始训练 ({quick_timesteps} 步)...")
        
        trained_model = agent.train_model(
            model=model,
            tb_log_name="quick_ppo",
            total_timesteps=quick_timesteps
        )
        
        print("✓ 训练完成!")
        
        # 测试交易
        print("开始回测...")
        env_kwargs_test = env_kwargs.copy()
        env_kwargs_test["print_verbosity"] = 0
        
        e_trade_gym = SignalTradingEnv(df=trade_data, **env_kwargs_test)
        
        df_account_value, df_actions = DRLAgent.DRL_prediction(
            model=trained_model, environment=e_trade_gym
        )
        
        # 计算结果
        initial_value = env_kwargs["initial_amount"]
        if not df_account_value.empty:
            final_value = df_account_value['account_value'].iloc[-1]
            total_return = (final_value - initial_value) / initial_value
            
            print("=" * 50)
            print("快速测试结果:")
            print(f"初始资金: {initial_value:,.0f}")
            print(f"最终资金: {final_value:,.0f}")
            print(f"总收益率: {total_return:.2%}")
            print(f"交易次数: {len(df_actions)}")
            
            # 简单的基准比较
            if not trade_data.empty:
                start_price = trade_data['close'].iloc[0]
                end_price = trade_data['close'].iloc[-1]
                benchmark_return = (end_price - start_price) / start_price
                print(f"基准收益率(买入持有): {benchmark_return:.2%}")
                print(f"超额收益: {(total_return - benchmark_return):.2%}")
            
        else:
            print("⚠️ 没有生成交易数据")
        
        return True
        
    except Exception as e:
        print(f"✗ 快速测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_environment_test():
    """运行环境基础测试"""
    print("=" * 50)
    print("运行环境基础测试...")
    
    try:
        # 创建简单测试数据
        test_data = pd.DataFrame({
            'time': ['2019-01-01', '2019-01-02', '2019-01-03'],
            'tic': ['600000.SH'] * 3,
            'close': [10.0, 10.5, 9.8],
            'macd': [0.1, 0.2, -0.1],
            'boll_ub': [11.0, 11.5, 10.8],
            'boll_lb': [9.0, 9.5, 8.8],
            'rsi_30': [50, 60, 40],
            'cci_30': [0, 10, -10],
            'dx_30': [30, 35, 25],
            'close_30_sma': [10.0, 10.2, 10.1],
            'close_60_sma': [10.0, 10.1, 10.05],
        })
        
        # 创建环境
        env_kwargs = {
            "initial_amount": 10000,
            "buy_cost_pct": 0.001,
            "sell_cost_pct": 0.001,
            "tech_indicator_list": ['macd', 'boll_ub', 'boll_lb', 'rsi_30', 'cci_30', 'dx_30', 'close_30_sma', 'close_60_sma'],
            "reward_config": {'method': 'information_ratio', 'return_weight': 1.0, 'risk_penalty_weight': 0.5, 'trade_quality_weight': 0.1, 'final_reward_weight': 2.0, 'benchmark': 'buy_hold'},
        }
        
        env = SignalTradingEnv(df=test_data, **env_kwargs)
        
        print(f"✓ 环境创建成功")
        print(f"✓ 动作空间: {env.action_space}")
        print(f"✓ 观测空间: {env.observation_space}")
        print(f"✓ 状态维度: {env.state_dim}")
        
        # 测试基本操作
        state = env.reset()
        print(f"✓ 初始状态形状: {np.array(state).shape}")
        
        # 执行几个动作
        for i, action in enumerate([1, 0, 1]):  # 买入, 卖出, 买入
            state, reward, done, info = env.step(action)
            print(f"✓ 步骤 {i+1}: 动作={action}, 奖励={reward:.4f}, 完成={done}")
            if done:
                break
        
        print("✓ 环境基础测试通过!")
        return True
        
    except Exception as e:
        print(f"✗ 环境测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("🚀 开始信号交易快速测试")
    print("=" * 50)
    
    # 1. 环境基础测试
    print("第 1 步: 环境基础测试")
    if not run_environment_test():
        print("环境测试失败，退出")
        return
    
    # 2. 数据准备
    print("\n第 2 步: 数据准备")
    train_data, trade_data = quick_data_preparation()
    
    if train_data is None or trade_data is None:
        print("数据准备失败，退出")
        return
    
    # 3. 快速训练测试
    print("\n第 3 步: 快速训练测试")
    success = quick_train_test(train_data, trade_data)
    
    if success:
        print("\n🎉 所有测试通过!")
        print("信号交易系统已成功开发并测试！")
        print("\n系统特点:")
        print("- 离散动作空间: 0=无持仓，1=多头持有")
        print("- 综合奖励函数: 考虑收益、风险、交易成本")
        print("- 增强状态空间: 包含持仓状态和实时风控指标")
        print("- 适用场景: 单标的择时交易策略")
    else:
        print("\n❌ 测试未完全通过，请检查环境配置")

if __name__ == "__main__":
    main() 