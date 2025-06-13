import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import random
import torch
import os
import pickle
from IPython import display

display.set_matplotlib_formats("svg")

def set_global_seed(seed=42):
    """设置全局随机种子以确保结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

# 设置随机种子
set_global_seed(42)

from meta import config
from meta.data_processor import DataProcessor
from main import check_and_make_directories
from meta.data_processors.tushare import Tushare, ReturnPlotter
from meta.env_stock_trading.env_signal_trading import SignalTradingEnv
from agents.stablebaselines3_models import DRLAgent
from meta.config import (
    DATA_SAVE_DIR,
    TRAINED_MODEL_DIR,
    TENSORBOARD_LOG_DIR,
    RESULTS_DIR,
    # INDICATORS,
    INDICATORS_OMK,
)

print("所有模块导入完成!")

# 创建必要的文件夹
check_and_make_directories([DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR])

# ================================
# 配置参数
# ================================

# 单股票标的
ticker_list = ["000905.XSHG"]  # 浦发银行作为测试标的

# 日期设置
TRAIN_START_DATE = "2015-01-01"
TRAIN_END_DATE = "2019-08-01"
TRADE_START_DATE = "2019-08-01"
TRADE_END_DATE = "2020-08-01"

TIME_INTERVAL = "1d"

# 奖励函数权重配置
REWARD_WEIGHTS = {
    'return': 1.0,        # 收益率权重
    'sharpe': 0.2,        # 夏普比率权重
    'max_drawdown': -1.0, # 最大回撤惩罚权重
    'volatility': -0.1    # 波动率惩罚权重
}

print(f"训练期间: {TRAIN_START_DATE} 到 {TRAIN_END_DATE}")
print(f"交易期间: {TRADE_START_DATE} 到 {TRADE_END_DATE}")
print(f"标的股票: {ticker_list}")

# ================================
# 数据下载和处理
# ================================

# finrl_data = pd.read_csv('../../data/filtered_regime2.csv')
finrl_data = pd.read_csv('/Users/qinchaoyi/workspace/Ultron/code/AI4Fin/FinRL-Meta/data/filtered_regime2.csv')

print(f"数据形状: {finrl_data.shape}")
print(f"数据列: {finrl_data.columns.tolist()}")

# ================================
# 数据分割
# ================================

train = finrl_data[finrl_data['time'] >= TRAIN_START_DATE]
train = train[train['time'] <= TRAIN_END_DATE]

trade = finrl_data[finrl_data['time'] >= TRADE_START_DATE]
trade = trade[trade['time'] <= TRADE_END_DATE]

# train.to_csv(rf'/Users/qinchaoyi/workspace/Ultron/code/AI4Fin/FinRL-Meta/datasets/train.csv')
print(f"训练数据形状: {train.shape}")
print(f"交易数据形状: {trade.shape}")
print(f"股票数量: {len(train.tic.unique())}")

# ================================
# 环境设置
# ================================

# 改进的奖励配置
reward_config_balanced = {
    'method': 'information_ratio',  # 使用信息比率方法
    'return_weight': 1.0,
    'risk_penalty_weight': 0.5,
    'trade_quality_weight': 0.1,
    'final_reward_weight': 2.0,
    'benchmark': 'buy_hold'
}

reward_config_multi_factor = {
    'method': 'multi_factor',  # 使用多因子方法
    'return_weight': 1.0,
    'risk_penalty_weight': 0.8,
    'trade_quality_weight': 0.15,
    'final_reward_weight': 1.5,
    'benchmark': 'buy_hold'
}

# 交易环境参数
env_kwargs = {
    "initial_amount": 1000000,
    "buy_cost_pct": 1.25e-3,
    "sell_cost_pct": 1.25e-3,
    "tech_indicator_list": config.INDICATORS_OMK,
    "turbulence_threshold": None,
    "make_plots": True,
    "print_verbosity": 10,
    "reward_config": reward_config_balanced,  # 使用新的reward_config
    "random_seed": 42,
}

# 创建训练环境
print("创建训练环境...")
e_train_gym = SignalTradingEnv(df=train, **env_kwargs)
env_train, _ = e_train_gym.get_sb_env()

print(f"动作空间: {e_train_gym.action_space}")
print(f"观测空间: {e_train_gym.observation_space}")
print(f"状态维度: {e_train_gym.state_dim}")

# ================================
# 模型训练
# ================================

def train_signal_models():
    """训练多个模型并比较性能"""
    
    models_config = {
        "PPO": {
            "learning_rate": 0.0003,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
        },
        "A2C": {
            "learning_rate": 0.0007,
            "n_steps": 5,
            "gamma": 0.99,
            "gae_lambda": 1.0,
            "ent_coef": 0.01,
            "vf_coef": 0.25,
        },
        # "DQN": {
        #     "learning_rate": 1e-3,
        #     "buffer_size": 50000,
        #     "learning_starts": 1000,
        #     "batch_size": 32,
        #     "tau": 1.0,
        #     "gamma": 0.99,
        #     "train_freq": 4,
        #     "gradient_steps": 1,
        #     "target_update_interval": 1000,
        # }
    }
    
    trained_models = {}
    
    for model_name, params in models_config.items():
        print(f"\n=== 训练 {model_name} 模型 ===")
        
        # 重新创建环境以避免状态干扰
        e_train_fresh = SignalTradingEnv(df=train, **env_kwargs)
        env_train_fresh, _ = e_train_fresh.get_sb_env()
        
        agent = DRLAgent(env=env_train_fresh)
        
        try:
            model = agent.get_model(model_name.lower(), model_kwargs=params, seed=42)
            trained_model = agent.train_model(
                model=model, 
                tb_log_name=f"{model_name.lower()}_signal_trading", 
                total_timesteps=50000
            )
            trained_models[model_name] = trained_model
            print(f"{model_name} 模型训练完成!")
            
        except Exception as e:
            print(f"{model_name} 模型训练失败: {e}")
            continue
    
    return trained_models

# 训练模型
print("开始模型训练...")
trained_models = train_signal_models()

# ================================
# 模型评估和回测
# ================================

def evaluate_model(model, model_name, trade_data):
    """评估单个模型"""
    print(f"\n=== 评估 {model_name} 模型 ===")
    
    # 创建交易环境
    env_kwargs_trade = env_kwargs.copy()
    env_kwargs_trade["print_verbosity"] = 1
    env_kwargs_trade["make_plots"] = True
    
    e_trade_gym = SignalTradingEnv(df=trade_data, **env_kwargs_trade)
    
    # 执行交易
    df_account_value, df_actions = DRLAgent.DRL_prediction(
        model=model, environment=e_trade_gym
    )
    
    # 保存结果
    df_account_value.to_csv(f"results/{model_name.lower()}_account_value.csv", index=False)
    df_actions.to_csv(f"results/{model_name.lower()}_actions.csv", index=False)
    
    # 计算性能指标
    if not df_account_value.empty:
        initial_value = env_kwargs["initial_amount"]
        final_value = df_account_value['account_value'].iloc[-1]
        total_return = (final_value - initial_value) / initial_value
        
        # 计算日收益率
        df_account_value['daily_return'] = df_account_value['account_value'].pct_change()
        daily_returns = df_account_value['daily_return'].dropna()
        
        # 计算风险指标
        sharpe_ratio = 0
        max_drawdown = 0
        volatility = 0
        
        if len(daily_returns) > 1:
            sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0
            volatility = daily_returns.std() * np.sqrt(252)
            
            # 计算最大回撤
            cumulative = (1 + daily_returns).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdown = (cumulative - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
        
        performance_stats = {
            'Model': model_name,
            'Total Return': f"{total_return:.4f}",
            'Final Value': f"{final_value:.2f}",
            'Sharpe Ratio': f"{sharpe_ratio:.4f}",
            'Max Drawdown': f"{max_drawdown:.4f}",
            'Volatility': f"{volatility:.4f}",
            'Total Trades': len(df_actions) if not df_actions.empty else 0
        }
        
        print("性能统计:")
        for key, value in performance_stats.items():
            print(f"  {key}: {value}")
            
        return df_account_value, df_actions, performance_stats
    
    return None, None, None

# 评估所有模型
print("\n开始模型评估...")
all_results = {}
performance_comparison = []

for model_name, model in trained_models.items():
    try:
        account_value, actions, stats = evaluate_model(model, model_name, trade)
        if stats:
            all_results[model_name] = {
                'account_value': account_value,
                'actions': actions,
                'stats': stats
            }
            performance_comparison.append(stats)
    except Exception as e:
        print(f"评估 {model_name} 模型时出错: {e}")

# ================================
# 结果比较和可视化
# ================================

if performance_comparison:
    print("\n=== 模型性能比较 ===")
    comparison_df = pd.DataFrame(performance_comparison)
    print(comparison_df.to_string(index=False))
    comparison_df.to_csv("results/model_performance_comparison.csv", index=False)

# 绘制比较图表
if len(all_results) > 1:
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(15, 10))
    
    # 资产曲线比较
    plt.subplot(2, 2, 1)
    for model_name, result in all_results.items():
        if result['account_value'] is not None:
            account_value = result['account_value']
            plt.plot(pd.to_datetime(account_value['date']), 
                    account_value['account_value'], 
                    label=model_name, linewidth=2)
    
    plt.title("Portfolio Value Comparison")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 动作比较
    plt.subplot(2, 2, 2)
    model_names = list(all_results.keys())
    total_trades = [len(all_results[name]['actions']) if all_results[name]['actions'] is not None else 0 
                   for name in model_names]
    plt.bar(model_names, total_trades)
    plt.title("Total Trades Comparison")
    plt.xlabel("Model")
    plt.ylabel("Number of Trades")
    plt.xticks(rotation=45)
    
    # 收益率比较
    plt.subplot(2, 2, 3)
    returns = []
    for name in model_names:
        if all_results[name]['account_value'] is not None:
            account_value = all_results[name]['account_value']
            initial_value = env_kwargs["initial_amount"]
            final_value = account_value['account_value'].iloc[-1]
            total_return = (final_value - initial_value) / initial_value
            returns.append(total_return)
        else:
            returns.append(0)
    
    colors = ['green' if r > 0 else 'red' for r in returns]
    plt.bar(model_names, returns, color=colors, alpha=0.7)
    plt.title("Total Return Comparison")
    plt.xlabel("Model")
    plt.ylabel("Total Return")
    plt.xticks(rotation=45)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # 基准比较（买入持有策略）
    plt.subplot(2, 2, 4)
    if not trade.empty:
        initial_price = trade['close'].iloc[0]
        final_price = trade['close'].iloc[-1]
        buy_hold_return = (final_price - initial_price) / initial_price
        
        strategy_returns = returns + [buy_hold_return]
        strategy_names = model_names + ['Buy & Hold']
        colors = ['blue'] * len(model_names) + ['orange']
        
        plt.bar(strategy_names, strategy_returns, color=colors, alpha=0.7)
        plt.title("Strategy vs Buy & Hold")
        plt.xlabel("Strategy")
        plt.ylabel("Total Return")
        plt.xticks(rotation=45)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        print(f"\n买入持有基准收益率: {buy_hold_return:.4f}")
    
    plt.tight_layout()
    plt.savefig("results/signal_trading_comparison.png", dpi=150, bbox_inches='tight')
    plt.show()

# ================================
# 保存最终结果
# ================================

# 单独保存训练好的模型（使用模型自己的save方法）
print("\n保存训练好的模型...")
for model_name, model in trained_models.items():
    try:
        model_path = f"{TRAINED_MODEL_DIR}/{model_name.lower()}_signal_trading"
        model.save(model_path)
        print(f"✓ {model_name} 模型已保存到: {model_path}")
    except Exception as e:
        print(f"✗ 保存 {model_name} 模型失败: {e}")

# 准备可序列化的结果数据（排除不可序列化的模型对象）
serializable_results = {
    'performance_comparison': performance_comparison,
    'trade_data_info': {
        'shape': trade.shape if not trade.empty else (0, 0),
        'columns': trade.columns.tolist() if not trade.empty else [],
        'date_range': [trade['time'].min(), trade['time'].max()] if not trade.empty and 'time' in trade.columns else [],
        'stock_symbols': trade['tic'].unique().tolist() if not trade.empty and 'tic' in trade.columns else []
    },
    'env_kwargs': env_kwargs,
    'model_names': list(trained_models.keys()),
    'training_completed': True,
    'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
}

# 添加账户价值数据（不包含模型对象）
if all_results:
    serializable_results['account_values'] = {}
    serializable_results['actions'] = {}
    
    for model_name, result in all_results.items():
        if result['account_value'] is not None:
            serializable_results['account_values'][model_name] = result['account_value'].to_dict('records')
        if result['actions'] is not None:
            serializable_results['actions'][model_name] = result['actions'].to_dict('records')

# 保存可序列化的结果
try:
    with open("results/signal_trading_final_results.pkl", "wb") as f:
        pickle.dump(serializable_results, f)
    print("✓ 可序列化结果已保存到: results/signal_trading_final_results.pkl")
except Exception as e:
    print(f"✗ 保存序列化结果失败: {e}")

# 另外保存为JSON格式（更通用）
try:
    import json
    # 转换为JSON兼容格式
    json_results = serializable_results.copy()
    
    # 将可能的numpy类型转换为Python原生类型
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    # 递归转换
    def recursive_convert(obj):
        if isinstance(obj, dict):
            return {k: recursive_convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [recursive_convert(v) for v in obj]
        else:
            return convert_numpy(obj)
    
    json_results = recursive_convert(json_results)
    
    with open("results/signal_trading_final_results.json", "w", encoding='utf-8') as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    print("✓ JSON格式结果已保存到: results/signal_trading_final_results.json")
except Exception as e:
    print(f"✗ 保存JSON结果失败: {e}")

print("\n=== 信号交易策略开发完成 ===")
print("结果已保存到以下位置:")
print("\n模型文件:")
for model_name in trained_models.keys():
    model_path = f"{TRAINED_MODEL_DIR}/{model_name.lower()}_signal_trading"
    print(f"- {model_path}.zip: {model_name} 训练好的模型")

print("\n结果数据:")
print("- results/signal_trading_final_results.pkl: 完整结果（pickle格式）")
print("- results/signal_trading_final_results.json: 完整结果（JSON格式）")
print("- results/model_performance_comparison.csv: 模型性能比较")
print("- results/signal_trading_comparison.png: 可视化比较图")
print("- results/*_account_value.csv: 各模型的资产曲线")
print("- results/*_actions.csv: 各模型的交易动作")

print("\n如何加载保存的模型:")
for model_name in trained_models.keys():
    print(f"- {model_name}: model = {model_name}(\"ppo\").load(\"{TRAINED_MODEL_DIR}/{model_name.lower()}_signal_trading\")")

# 显示最佳模型
if performance_comparison:
    best_model = max(performance_comparison, key=lambda x: float(x['Total Return']))
    print(f"\n最佳模型: {best_model['Model']}")
    print(f"总收益率: {best_model['Total Return']}")
    print(f"夏普比率: {best_model['Sharpe Ratio']}")
    print(f"最大回撤: {best_model['Max Drawdown']}") 