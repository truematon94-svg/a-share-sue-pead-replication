"""
SUE2 因子检验：定期财报 + 业绩快报，全A，月频，2009.12–2020.01。
输出同 SUE1，用于对照研报 SUE2：RankIC 4.4%、T 10.37、月胜率 85.1%、多空年化 19.31%。

与 SUE1 的区别：
- 多取一张"业绩快报"表(STK_PERFORMANCE_LETTERS)，它的归母净利润比定期财报更早披露。
- 快报主要覆盖 半年报(Q2) 和 年报(Q4)，所以 Q2/Q4 能比 SUE1 早几个月知道 → 因子更及时。
- 实现：把定期财报和快报的"累计归母净利润"合成一张事件表(各带 pub_date)，
  每个月按 pub_date<=信号日 过滤后，对每个(股票,报告期)取"最新披露"那条(定期晚→优先，快报早→先顶)。
"""

from jqdata import *
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

START = '2009-12-01'
END = '2020-01-23'
FETCH_FROM = '2006-01-01'
CHUNK = 40

# ---- 1. 信号日：每月最后一个交易日 ----
days = pd.DatetimeIndex(get_trade_days(start_date=START, end_date=END))
signal_dates = pd.DatetimeIndex(
    pd.Series(days, index=days).groupby(days.to_period('M')).max().values).sort_values()
print('信号日数量:', len(signal_dates), signal_dates[0].date(), '->', signal_dates[-1].date())

# ---- 2. 全市场股票(含退市) ----
sec = get_all_securities(types=['stock'])
all_codes = sec.index.tolist()
print('股票总数(含退市):', len(all_codes))

# ---- 3. 定期财报（利润表）累计归母净利润 ----
def fetch_income(codes):
    frames = []
    for i in range(0, len(codes), CHUNK):
        chunk = codes[i:i + CHUNK]
        q = query(finance.STK_INCOME_STATEMENT.code,
                  finance.STK_INCOME_STATEMENT.pub_date,
                  finance.STK_INCOME_STATEMENT.end_date,
                  finance.STK_INCOME_STATEMENT.report_type,
                  finance.STK_INCOME_STATEMENT.np_parent_company_owners).filter(
            finance.STK_INCOME_STATEMENT.pub_date >= FETCH_FROM,
            finance.STK_INCOME_STATEMENT.pub_date <= END,
            finance.STK_INCOME_STATEMENT.code.in_(chunk))
        frames.append(finance.run_query(q))
        if (i // CHUNK) % 20 == 0:
            print('  取定期财报 %d/%d' % (i + len(chunk), len(codes)))
    return pd.concat(frames, ignore_index=True)


income = fetch_income(all_codes)
income = income[income['report_type'] == 0].copy()
income = income.rename(columns={'np_parent_company_owners': 'cum_np'})
print('定期财报行数(report_type=0):', len(income))

# ---- 4. 业绩快报累计归母净利润 ----
def fetch_express(codes):
    frames = []
    for i in range(0, len(codes), CHUNK):
        chunk = codes[i:i + CHUNK]
        q = query(finance.STK_PERFORMANCE_LETTERS.code,
                  finance.STK_PERFORMANCE_LETTERS.pub_date,
                  finance.STK_PERFORMANCE_LETTERS.end_date,
                  finance.STK_PERFORMANCE_LETTERS.np_parent_company_owners).filter(
            finance.STK_PERFORMANCE_LETTERS.pub_date >= FETCH_FROM,
            finance.STK_PERFORMANCE_LETTERS.pub_date <= END,
            finance.STK_PERFORMANCE_LETTERS.code.in_(chunk))
        frames.append(finance.run_query(q))
        if (i // CHUNK) % 20 == 0:
            print('  取业绩快报 %d/%d' % (i + len(chunk), len(codes)))
    return pd.concat(frames, ignore_index=True)


express = fetch_express(all_codes)
express = express.rename(columns={'np_parent_company_owners': 'cum_np'})
print('业绩快报行数:', len(express))

# ---- 5. 合并成"累计归母净利润事件表" ----
cum_events = pd.concat([
    income[['code', 'pub_date', 'end_date', 'cum_np']],
    express[['code', 'pub_date', 'end_date', 'cum_np']],
], ignore_index=True)
cum_events['pub_date'] = pd.to_datetime(cum_events['pub_date'])
cum_events['end_date'] = pd.to_datetime(cum_events['end_date'])
cum_events['month'] = cum_events['end_date'].dt.month
cum_events = cum_events[cum_events['month'].isin([3, 6, 9, 12])].copy()
print('合并事件表行数:', len(cum_events))

# ---- 6. 价格：全市场前复权收盘（分块） ----
def fetch_close(codes):
    parts = []
    for i in range(0, len(codes), 500):
        chunk = codes[i:i + 500]
        p = get_price(chunk, start_date=START, end_date=END,
                      frequency='daily', fields=['close'], fq='pre', skip_paused=False)
        if isinstance(p, pd.Panel):
            p = p['close']          # items=字段 → 日期×代码
        else:
            p = p[['close']].rename(columns={'close': chunk[0]})
        parts.append(p)
        if (i // 500) % 5 == 0:
            print('  取价格 %d/%d' % (i + len(chunk), len(codes)))
    return pd.concat(parts, axis=1, sort=True)


close = fetch_close(all_codes)
print('价格矩阵 shape:', close.shape)
signal_dates = signal_dates[signal_dates.isin(close.index)]


# ---- 7. SUE 计算 ----
def sue_from_series(vals):
    v = np.asarray(vals, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) < 13:
        return np.nan
    t = -1
    diffs = [v[t - j] - v[t - j - 4] for j in range(1, 9)]
    delta = np.mean(diffs)
    sigma = np.std([d - delta for d in diffs], ddof=1)
    if not np.isfinite(sigma) or sigma == 0:
        return np.nan
    return (v[t] - (v[t - 4] + delta)) / sigma


# ---- 8. 逐月：点前过滤 -> 最新披露 -> 单季度 -> SUE -> IC ----
ic_rows = []
group_rows = []

for i in range(len(signal_dates) - 1):
    D = signal_dates[i]
    Dn = signal_dates[i + 1]
    avail = cum_events[cum_events['pub_date'] <= D]   # 点前可用的累计值（定期+快报）
    best = avail.sort_values('pub_date').drop_duplicates(['code', 'end_date'], keep='last')
    # 向量化算单季度：groupby.shift 取代原来的 groupby.apply（快很多）
    best = best.sort_values(['code', 'end_date'])
    prev = best.groupby('code')['cum_np'].shift(1)
    best['single'] = np.where(best['month'].values == 3,
                              best['cum_np'].values,
                              best['cum_np'].values - prev.values)
    sue = best.groupby('code')['single'].apply(sue_from_series).dropna()
    if len(sue) < 50:
        continue
    fwd = close.loc[Dn] / close.loc[D] - 1
    df = pd.concat([sue.rename('sue'), fwd.rename('fwd')], axis=1).dropna()
    if len(df) < 50:
        continue
    ic_rows.append((D, spearmanr(df['sue'], df['fwd'])[0]))
    g = pd.qcut(df['sue'].rank(method='first'), 10, labels=False, duplicates='drop')
    group_rows.append(df.groupby(g)['fwd'].mean())

# ---- 9. 汇总 ----
ic = pd.Series(dict(ic_rows)).dropna()
print('\n===== SUE2 结果 =====')
print('月数:', len(ic))
print('RankIC 均值: %.4f' % ic.mean())
print('RankIC T值: %.2f' % (ic.mean() / ic.std() * np.sqrt(len(ic))))
print('月胜率: %.2f%%' % ((ic > 0).mean() * 100))

gret = pd.DataFrame(group_rows).mean()
print('\n各组月均收益(%), 0=不达预期 ... 9=超预期:')
print((gret * 100).round(2))
print('\n多空(9-0)月均收益: %.3f%%' % ((gret.iloc[9] - gret.iloc[0]) * 100))
print('多空约年化: %.2f%%' % ((gret.iloc[9] - gret.iloc[0]) * 12 * 100))
