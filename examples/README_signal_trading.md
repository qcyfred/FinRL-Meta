# 信号交易策略系统

## 概述

本系统是基于强化学习的单标的择时交易策略，专门设计用于处理单一资产的买入/卖出决策。系统将连续的持仓量决策简化为离散的信号决策（多头持有/无持仓），更适合实际的择时交易场景。

## 系统特点

### 1. 离散化动作空间
- **传统方式**: 连续的股票数量（可以买入任意数量股票）
- **信号交易**: 离散的持仓状态（0=无持仓，1=多头持有）
- **优势**: 更符合实际择时策略的决策模式

### 2. 综合奖励函数
系统采用多因子奖励函数，综合考虑：
- **收益率**: 基础收益表现
- **夏普比率**: 风险调整后收益
- **最大回撤**: 风险控制指标
- **波动率**: 稳定性指标
- **交易成本**: 实际交易成本惩罚

### 3. 增强状态空间
状态信息包括：
- 当前现金和股票持仓
- 持仓状态（0/1）
- 股票价格和技术指标
- 实时风控指标（最大回撤、当前回撤、波动率、夏普比率）

### 4. 多策略配置
预定义了多种策略类型：
- **保守策略**: 重视风控，适合风险厌恶者
- **平衡策略**: 收益与风控并重
- **激进策略**: 追求高收益
- **动量策略**: 快速响应趋势变化
- **均值回归策略**: 适合震荡市场

## 文件结构

```
examples/
├── signal_trading_main.py      # 完整的训练和回测主程序
├── quick_signal_test.py        # 快速测试脚本
├── signal_config.py            # 策略配置文件
└── README_signal_trading.md    # 本文档

meta/env_stock_trading/
└── env_signal_trading.py       # 信号交易环境
```

## 核心组件

### 1. SignalTradingEnv (信号交易环境)

**文件**: `meta/env_stock_trading/env_signal_trading.py`

**核心特性**:
- 继承自 OpenAI Gym，兼容强化学习框架
- 离散动作空间：`Discrete(2)` (0=卖出/持有, 1=买入/持有)
- 增强状态空间：包含价格、技术指标、持仓状态、风控指标
- 多因子奖励函数：综合收益、风险、交易成本

**关键方法**:
```python
def _go_long(self):     # 执行多头操作（满仓买入）
def _go_neutral(self):  # 执行平仓操作（卖出所有）
def _calculate_reward(self, prev_value, current_value):  # 计算综合奖励
def _calculate_risk_indicators(self):  # 计算风控指标
```

### 2. 策略配置系统

**文件**: `examples/signal_config.py`

**功能**:
- 定义不同风险偏好的奖励权重
- 配置不同算法的超参数
- 预定义策略组合
- 提供策略选择和配置接口

**预定义策略**:
```python
STRATEGY_COMBINATIONS = {
    "conservative_portfolio": {...},    # 保守策略
    "balanced_portfolio": {...},        # 平衡策略  
    "aggressive_portfolio": {...},      # 激进策略
    "momentum_strategy": {...},         # 动量策略
    "mean_reversion_strategy": {...}    # 均值回归策略
}
```

### 3. 主训练程序

**文件**: `examples/signal_trading_main.py`

**功能**:
- 数据下载和预处理
- 多模型训练（PPO、A2C、DQN）
- 模型评估和比较
- 结果可视化和保存

### 4. 快速测试工具

**文件**: `examples/quick_signal_test.py`

**功能**:
- 环境基础功能测试
- 模拟数据生成（用于快速测试）
- 简化版训练流程
- 基本性能验证

## 使用指南

### 1. 快速开始

```bash
# 运行快速测试（推荐首次使用）
cd examples
python quick_signal_test.py
```

这将：
- 测试环境基础功能
- 生成或下载测试数据  
- 进行快速训练和回测
- 显示基本结果

### 2. 完整训练

```bash
# 运行完整的训练和回测
python signal_trading_main.py
```

这将：
- 下载真实市场数据
- 训练多个强化学习模型
- 进行详细的性能评估
- 生成完整的分析报告

### 3. 自定义策略

```python
# 使用预定义策略
from signal_config import get_strategy_config

strategy_config = get_strategy_config("momentum_strategy")
print(strategy_config)

# 查看所有可用策略
from signal_config import list_available_strategies
strategies = list_available_strategies()
for strategy in strategies:
    print(f"{strategy['name']}: {strategy['description']}")
```

### 4. 自定义环境参数

```python
# 自定义奖励权重
custom_rewards = {
    'return': 1.5,        # 更重视收益
    'sharpe': 0.0,        # 忽略夏普比率
    'max_drawdown': -2.0, # 严格控制回撤
    'volatility': -0.1    # 轻微惩罚波动率
}

# 创建环境
env = SignalTradingEnv(
    df=data,
    initial_amount=1000000,
    buy_cost_pct=0.001,
    sell_cost_pct=0.001,
    reward_weights=custom_rewards,
    tech_indicator_list=['macd', 'rsi_30', 'boll_ub', 'boll_lb']
)
```

## 性能指标

系统提供全面的性能评估指标：

### 主要指标
- **总收益率**: 策略的总体收益表现
- **夏普比率**: 风险调整后收益
- **最大回撤**: 最大的峰值到谷值下跌
- **波动率**: 收益的标准差（年化）
- **交易次数**: 总的买卖操作数量

### 次要指标
- **胜率**: 盈利交易占比
- **平均交易收益**: 每笔交易的平均收益
- **盈亏比**: 平均盈利与平均亏损的比率
- **卡尔玛比率**: 收益率与最大回撤的比率
- **索提诺比率**: 考虑下行风险的收益比率

## 与传统方法的对比

### 传统多股票环境 (env_stocktrading_omk.py)
- 动作空间：连续，每只股票的买卖数量
- 适用场景：投资组合管理
- 复杂度：高（需要处理多资产配置）

### 信号交易环境 (env_signal_trading.py)  
- 动作空间：离散，买入/卖出信号
- 适用场景：单标的择时交易
- 复杂度：适中（专注于时机选择）

### 优势对比
1. **决策简化**: 从"买多少"简化为"买不买"
2. **实用性强**: 更符合实际择时策略的使用场景
3. **风控完善**: 集成多种风险控制指标
4. **易于理解**: 交易逻辑更直观
5. **可扩展性**: 易于添加新的技术指标和风控措施

## 扩展方向

### 1. 多时间框架
- 支持不同时间周期（日线、小时线、分钟线）
- 多时间框架信号融合

### 2. 更多资产类型
- 期货合约择时
- 外汇交易信号
- 加密货币交易

### 3. 高级策略
- 基于volatility targeting的仓位管理
- 动态止损和止盈
- 机器学习特征工程

### 4. 实盘连接
- 实盘API接口
- 实时数据流处理
- 订单执行优化

## 注意事项

1. **数据质量**: 确保使用高质量的历史数据
2. **过拟合**: 避免在训练集上过度优化
3. **市场变化**: 定期重新训练模型适应市场变化
4. **交易成本**: 考虑真实的交易成本和滑点
5. **风险管理**: 设置合理的风险控制参数

## 贡献指南

欢迎提交改进建议和代码贡献：

1. Fork 项目仓库
2. 创建新的功能分支
3. 实现新功能或修复bug
4. 添加相应的测试
5. 提交 Pull Request

## 常见问题

### Q: 为什么选择离散动作空间？
A: 离散动作空间更符合实际择时交易的决策模式，大多数择时策略关注的是"何时买入/卖出"而不是"买入多少"。

### Q: 如何选择合适的策略？
A: 根据您的风险偏好选择：
- 保守投资者：选择 `conservative_portfolio`
- 均衡投资者：选择 `balanced_portfolio`  
- 激进投资者：选择 `aggressive_portfolio`

### Q: 如何添加新的技术指标？
A: 在数据预处理阶段添加新指标，然后在环境的 `tech_indicator_list` 中包含该指标即可。

### Q: 系统是否支持做空？
A: 当前版本专注于多头择时，未来版本将考虑添加做空功能。

---

*如有问题或建议，请通过 GitHub Issues 联系我们。* 