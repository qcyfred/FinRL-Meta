"""Use WonderTrader Simulator""" # 使用WonderTrader模拟器

"""https://github.com/drlgistics/Wt4ElegantRL"""
from meta.env_future_trading.wt4elegantrl.envs_simple_cta import SimpleCTAEnv
from meta.env_future_trading.wt4elegantrl.envs_simple_cta import (
    SimpleCTASubProcessEnv,
)


class FutureTradingEnv_WtTrainer(SimpleCTASubProcessEnv):
    """
    期货交易训练环境 - Futures Trading Training Environment
    
    交易模式 (Trading Mode):
    - CTA(商品交易顾问)策略交易 - CTA (Commodity Trading Advisor) strategy trading
    - 期货合约的多空双向交易 - Long/short bidirectional trading of futures contracts
    - 基于技术分析的趋势跟踪策略 - Trend following strategies based on technical analysis
    - 支持多品种期货同时交易 - Supports simultaneous trading of multiple futures varieties
    
    应用场景 (Application Scenarios):
    - 商品期货的CTA策略开发 - CTA strategy development for commodity futures
    - 期货市场的趋势跟踪交易 - Trend following trading in futures markets
    - 期货套利和对冲策略 - Futures arbitrage and hedging strategies
    - 量化期货投资管理 - Quantitative futures investment management
    
    核心特点 (Key Features):
    - 基于WonderTrader的高性能模拟器 - Based on WonderTrader's high-performance simulator
    - 支持多进程并行训练 - Supports multi-process parallel training
    - 真实的期货交易规则和成本模拟 - Realistic futures trading rules and cost simulation
    - 灵活的时间范围配置 - Flexible time range configuration
    """
    env_num = 1  # 环境数量 - Number of environments
    max_step = 1500  # 最大步数 - Maximum steps
    if_discrete = False  # 连续动作空间 - Continuous action space

    @property
    def state_dim(self):
        """状态空间维度 - State space dimension"""
        return self.observation_space.shape[0]

    @property
    def action_dim(self):
        """动作空间维度 - Action space dimension"""
        # if len(self.action_space.shape) > 0 else 10
        return self.action_space.shape[0]

    def __init__(self):
        """初始化训练环境 - Initialize training environment"""
        super().__init__(
            **{
                # 'time_start': 202108301600, # 开始时间 - Start time
                # 'time_end': 202108311600,   # 结束时间 - End time
                "time_range": (  # 时间范围列表 - Time range list
                    # (201901011600, 202101011600), # 2019-2021年数据 - 2019-2021 data
                    # (201901011600, 201906301600), # 2019年上半年 - First half of 2019
                    # (201906301600, 202001011600), # 2019年下半年 - Second half of 2019
                    # (202001011600, 202006301600), # 2020年上半年 - First half of 2020
                    # (202006301600, 202101011600), # 2020年下半年 - Second half of 2020
                    # (201812311600, 201901311600), # 2018年12月-2019年1月 - Dec 2018 - Jan 2019
                    # (201901311600, 201902311600), # 2019年1月-2月 - Jan-Feb 2019
                    # (201902311600, 201903311600), # 2019年2月-3月 - Feb-Mar 2019
                    # (201903311600, 201904311600), # 2019年3月-4月 - Mar-Apr 2019
                    # (201904311600, 201905311600), # 2019年4月-5月 - Apr-May 2019
                    # (201905311600, 201906311600), # 2019年5月-6月 - May-Jun 2019
                    # (201906311600, 201907311600), # 2019年6月-7月 - Jun-Jul 2019
                    # (201907311600, 201908311600), # 2019年7月-8月 - Jul-Aug 2019
                    # (201908311600, 201909311600), # 2019年8月-9月 - Aug-Sep 2019
                    # (201909311600, 201910311600), # 2019年9月-10月 - Sep-Oct 2019
                    # (201910311600, 201911311600), # 2019年10月-11月 - Oct-Nov 2019
                    # (201911311600, 201912311600), # 2019年11月-12月 - Nov-Dec 2019
                    # (201912311600, 202001311600), # 2019年12月-2020年1月 - Dec 2019 - Jan 2020
                    # (202001311600, 202002311600), # 2020年1月-2月 - Jan-Feb 2020
                    # (202002311600, 202003311600), # 2020年2月-3月 - Feb-Mar 2020
                    # (202003311600, 202004311600), # 2020年3月-4月 - Mar-Apr 2020
                    # (202004311600, 202005311600), # 2020年4月-5月 - Apr-May 2020
                    # (202005311600, 202006311600), # 2020年5月-6月 - May-Jun 2020
                    # (202006311600, 202007311600), # 2020年6月-7月 - Jun-Jul 2020
                    # (202007311600, 202008311600), # 2020年7月-8月 - Jul-Aug 2020
                    # (202008311600, 202009311600), # 2020年8月-9月 - Aug-Sep 2020
                    (202009311600, 202010311600),  # 2020年9月-10月 - Sep-Oct 2020
                    (202010311600, 202011311600),  # 2020年10月-11月 - Oct-Nov 2020
                    (202011311600, 202012311600),  # 2020年11月-12月 - Nov-Dec 2020
                ),
                "slippage": 0,  # 滑点设置 - Slippage setting
                "mode": 1,      # 运行模式 - Running mode
            }
        )


class FutureTradingEnv_WtEvaluator(SimpleCTASubProcessEnv):
    """
    期货交易评估环境 - Futures Trading Evaluation Environment
    
    用于评估训练好的期货交易策略的性能表现 - Used to evaluate the performance of trained futures trading strategies
    包含更长的时间序列和更多的市场周期 - Contains longer time series and more market cycles
    """
    env_num = 1  # 环境数量 - Number of environments
    max_step = 1500  # 最大步数 - Maximum steps
    if_discrete = False  # 连续动作空间 - Continuous action space

    @property
    def state_dim(self):
        """状态空间维度 - State space dimension"""
        return self.observation_space.shape[0]

    @property
    def action_dim(self):
        """动作空间维度 - Action space dimension"""
        # if len(self.action_space.shape) > 0 else 10
        return self.action_space.shape[0]

    def __init__(self):  # mode=3可以打开详细调试模式 - mode=3 can enable detailed debugging mode
        """初始化评估环境 - Initialize evaluation environment"""
        super().__init__(
            **{
                "time_range": (  # 评估时间范围(更长期的数据) - Evaluation time range (longer-term data)
                    # (202101011600, 202106301600), # 2021年上半年 - First half of 2021
                    # (201701011600, 201706301600), # 2017年上半年 - First half of 2017
                    # (201706301600, 201801011600), # 2017年下半年 - Second half of 2017
                    # (201801011600, 201806301600), # 2018年上半年 - First half of 2018
                    # (201806301600, 201901011600), # 2018年下半年 - Second half of 2018
                    (202012311600, 202101311600),  # 2020年12月-2021年1月 - Dec 2020 - Jan 2021
                    (202101311600, 202102311600),  # 2021年1月-2月 - Jan-Feb 2021
                    (202102311600, 202103311600),  # 2021年2月-3月 - Feb-Mar 2021
                    (202103311600, 202104311600),  # 2021年3月-4月 - Mar-Apr 2021
                    (202104311600, 202105311600),  # 2021年4月-5月 - Apr-May 2021
                    (202105311600, 202106311600),  # 2021年5月-6月 - May-Jun 2021
                    (201612311600, 201701311600),  # 2016年12月-2017年1月 - Dec 2016 - Jan 2017
                    (201701311600, 201702311600),  # 2017年1月-2月 - Jan-Feb 2017
                    (201702311600, 201703311600),  # 2017年2月-3月 - Feb-Mar 2017
                    (201703311600, 201704311600),  # 2017年3月-4月 - Mar-Apr 2017
                    (201704311600, 201705311600),  # 2017年4月-5月 - Apr-May 2017
                    (201705311600, 201706311600),  # 2017年5月-6月 - May-Jun 2017
                    (201706311600, 201707311600),  # 2017年6月-7月 - Jun-Jul 2017
                    (201707311600, 201708311600),  # 2017年7月-8月 - Jul-Aug 2017
                    (201708311600, 201709311600),  # 2017年8月-9月 - Aug-Sep 2017
                    (201709311600, 201710311600),  # 2017年9月-10月 - Sep-Oct 2017
                    (201710311600, 201711311600),  # 2017年10月-11月 - Oct-Nov 2017
                    (201711311600, 201712311600),  # 2017年11月-12月 - Nov-Dec 2017
                    (201712311600, 201801311600),  # 2017年12月-2018年1月 - Dec 2017 - Jan 2018
                    (201801311600, 201802311600),  # 2018年1月-2月 - Jan-Feb 2018
                    (201802311600, 201803311600),  # 2018年2月-3月 - Feb-Mar 2018
                    (201803311600, 201804311600),  # 2018年3月-4月 - Mar-Apr 2018
                    (201804311600, 201805311600),  # 2018年4月-5月 - Apr-May 2018
                    (201805311600, 201806311600),  # 2018年5月-6月 - May-Jun 2018
                    (201806311600, 201807311600),  # 2018年6月-7月 - Jun-Jul 2018
                    (201807311600, 201808311600),  # 2018年7月-8月 - Jul-Aug 2018
                    (201808311600, 201809311600),  # 2018年8月-9月 - Aug-Sep 2018
                    (201809311600, 201810311600),  # 2018年9月-10月 - Sep-Oct 2018
                    (201810311600, 201811311600),  # 2018年10月-11月 - Oct-Nov 2018
                    (201811311600, 201812311600),  # 2018年11月-12月 - Nov-Dec 2018
                ),
                "slippage": 0,  # 滑点设置 - Slippage setting
                "mode": 1,      # 运行模式 - Running mode
            }
        )


class FutureTradingEnv_WtTester(SimpleCTAEnv):
    """
    期货交易测试环境 - Futures Trading Testing Environment
    
    用于最终测试训练好的期货交易策略 - Used for final testing of trained futures trading strategies
    单进程环境，适合策略的最终验证和部署前测试 - Single-process environment, suitable for final validation and pre-deployment testing
    """
    env_num = 1  # 环境数量 - Number of environments
    max_step = 1500  # 最大步数 - Maximum steps
    if_discrete = False  # 连续动作空间 - Continuous action space

    @property
    def state_dim(self):
        """状态空间维度 - State space dimension"""
        return self.observation_space.shape[0]

    @property
    def action_dim(self):
        """动作空间维度 - Action space dimension"""
        # if len(self.action_space.shape) > 0 else 10
        return self.action_space.shape[0]

    def __init__(self):  # mode=3可以打开详细调试模式 - mode=3 can enable detailed debugging mode
        """初始化测试环境 - Initialize testing environment"""
        super().__init__(
            **{
                "time_range": (  # 测试时间范围 - Testing time range
                    # (202101011600, 202106301600), # 2021年上半年 - First half of 2021
                    # (201701011600, 201706301600), # 2017年上半年 - First half of 2017
                    # (201706301600, 201801011600), # 2017年下半年 - Second half of 2017
                    # (201801011600, 201806301600), # 2018年上半年 - First half of 2018
                    # (201806301600, 201901011600), # 2018年下半年 - Second half of 2018
                    (202012311600, 202101311600),  # 2020年12月-2021年1月测试期 - Dec 2020 - Jan 2021 testing period
                ),
                "slippage": 0,  # 滑点设置 - Slippage setting
                "mode": 2,      # 测试模式 - Testing mode
            }
        )
