"""
聚宽策略回测：SUE × 反转 双变量分组多头（L4D0）月频调仓策略。

策略逻辑（对应财通证券"星火"多因子专题报告(十一)）：
  - 每月选股：SUE（业绩超预期，SUE3=定期财报+快报+预告）× Ret（反转，前1个月收益）双变量分组，
    取 L4D0 组合 = SUE 最高层 ∩ Ret 最低组 = "业绩超预期 + 前期超跌"。
  - 月频调仓：每月第一个交易日开盘，按上一月末算出的持仓等权调仓。

持仓 JSON 生成方式（研究环境）：
  见 notebooks/04_ch5_reversal_fusion.py —— 逐月算 L4D0，输出 {日期: [6位代码,...]}，
  代码去后缀（本策略用 normalize_code 还原）。

已知问题（容量）：
  持仓为全 L4D0（每月 50~120 只），资金等权分摊后：
    - 高价股买不到 100 股（A股最低 1 手）→ "开仓数量不能小于 100"
    - 先买的把资金占满 → "剩余可用资金不足"
  修法：
    1) 研究代码限制持仓数 head(30)
    2) 本策略加现金缓冲（把 weight 的 1.0 改成 0.95，见 rebalance 内注释）
  报错属容量问题，不影响"因子有效"结论。

回测区间：2010-02-01 ~ 2020-01-23（第一份持仓是 2010-01-29 算出的，2 月开始持有）。
"""

from jqdata import *
import json

# 把研究环境生成的持仓 JSON 粘到这里（'''...''' 之间）
HOLDINGS_JSON = '''把JSON粘到这里'''


def initialize(context):
    set_benchmark('000300.XSHG')
    set_option('use_real_price', True)
    set_option('avoid_future_data', True)
    set_order_cost(OrderCost(open_tax=0, close_tax=0.001,
                             open_commission=0.0003, close_commission=0.0003,
                             close_today_commission=0, min_commission=5), type='stock')
    set_slippage(PriceRelatedSlippage(0.002))
    log.set_level('order', 'error')

    # 读持仓 JSON，6 位代码还原为带交易所后缀的完整代码
    h = json.loads(HOLDINGS_JSON)
    g.holdings = {d: [normalize_code(c) for c in codes] for d, codes in h.items()}

    # 每月第一个交易日开盘调仓，用上一月末的持仓（无未来函数）
    run_monthly(rebalance, monthday=1, time='open', reference_security='000300.XSHG')


def rebalance(context):
    date = context.previous_date.strftime('%Y-%m-%d')
    targets = g.holdings.get(date, [])
    if not targets:
        return
    targets = set(targets)
    current = get_current_data()
    held = set(context.portfolio.positions.keys())

    # 先卖出不在目标池的
    for code in list(held - targets):
        d = current[code]
        if not d.paused and d.last_price > d.low_limit:
            order_target_value(code, 0)

    # 等权买入目标池（改 0.95 可留 5% 现金缓冲，减少"资金不足"报错）
    weight = context.portfolio.total_value / len(targets)
    for code in targets:
        d = current[code]
        if d.paused or d.last_price >= d.high_limit:
            continue
        order_target_value(code, weight)

    log.info('%s 目标 %d 只' % (date, len(targets)))
