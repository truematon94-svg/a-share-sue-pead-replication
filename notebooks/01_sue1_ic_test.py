"""
SUE1 因子完整检验：全A，月频，2009.12–2020.01，只用定期财报(利润表)的归母净利润。
输出：RankIC 均值 / T值 / 月胜率 + 10组收益 + 多空(最高组-最低组)。

对照目标(研报 SUE1，行业+市值中性化后)：RankIC 4.2%、T 9.98、月胜率 82.6%、多空年化 18.48%。

本版说明：
- 防未来函数：财报只取 pub_date<=信号日；单季度值在其报告披露日(pub_date)后才可用。
- 股票池用 get_all_securities 全量(含退市)，避免幸存者偏差。
"""

from jqdata import *
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

START = '2009-12-01'
END = '2020-01-23'
FETCH_FROM = '2006-01-01'   # 财报起点，给最早信号日留足 12 季历史
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

# ---- 3. 一次性取利润表 ----
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
            print('  取财报 %d/%d' % (i + len(chunk), len(codes)))
    return pd.concat(frames, ignore_index=True)


raw = fetch_income(all_codes)
print('财报原始行数:', len(raw))
raw = raw[raw['report_type'] == 0].copy()
raw['pub_date'] = pd.to_datetime(raw['pub_date'])
raw['end_date'] = pd.to_datetime(raw['end_date'])
raw = raw.sort_values(['code', 'end_date', 'pub_date']).drop_duplicates(['code', 'end_date'], keep='last')
print('report_type==0 去重后行数:', len(raw))

# ---- 4. 累计 -> 单季度（相邻做差，Q1 不差） ----
raw['month'] = raw['end_date'].dt.month
raw = raw[raw['month'].isin([3, 6, 9, 12])].copy()


def build_single(g):
    g = g.sort_values('end_date')
    prev = g['np_parent_company_owners'].shift(1)
    single = np.where(g['month'].values == 3, g['np_parent_company_owners'].values,
                      g['np_parent_company_owners'].values - prev.values)
    return pd.DataFrame({'code': g['code'].values,
                         'qdate': g['end_date'].values,
                         'pub_date': g['pub_date'].values,
                         'q_np': single})


sq = raw.groupby('code', group_keys=False).apply(build_single).reset_index(drop=True)
print('单季度表行数:', len(sq))

# ---- 5. 价格：全市场前复权收盘（分块） ----
def fetch_close(codes):
    parts = []
    for i in range(0, len(codes), 500):
        chunk = codes[i:i + 500]
        p = get_price(chunk, start_date=START, end_date=END,
                      frequency='daily', fields=['close'], fq='pre', skip_paused=False)
        # 老版聚宽 get_price 多标的返回 Panel（items=字段, major_axis=日期, minor_axis=代码）
        if isinstance(p, pd.Panel):
            p = p['close']          # 取 'close' 字段 → 日期×代码
        else:
            p = p[['close']].rename(columns={'close': chunk[0]})
        parts.append(p)
        if (i // 500) % 5 == 0:
            print('  取价格 %d/%d' % (i + len(chunk), len(codes)))
    return pd.concat(parts, axis=1, sort=True)


close = fetch_close(all_codes)
print('价格矩阵 shape:', close.shape)
signal_dates = signal_dates[signal_dates.isin(close.index)]


# ---- 6. SUE 计算 ----
def sue_from_series(vals):
    v = np.asarray(vals, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) < 13:            # 需要 t 及往前 12 个季度
        return np.nan
    t = -1
    diffs = [v[t - j] - v[t - j - 4] for j in range(1, 9)]
    delta = np.mean(diffs)
    sigma = np.std([d - delta for d in diffs], ddof=1)
    if not np.isfinite(sigma) or sigma == 0:
        return np.nan
    return (v[t] - (v[t - 4] + delta)) / sigma


# ---- 7. 逐月：SUE + 未来月收益 + IC + 分组 ----
ic_rows = []
group_rows = []

for i in range(len(signal_dates) - 1):
    D = signal_dates[i]
    Dn = signal_dates[i + 1]
    avail = sq[sq['pub_date'] <= D]   # 点前可用的单季度
    sue = avail.groupby('code').apply(
        lambda g: sue_from_series(g.sort_values('qdate')['q_np'].values)).dropna()
    if len(sue) < 50:
        continue
    fwd = close.loc[Dn] / close.loc[D] - 1
    df = pd.concat([sue.rename('sue'), fwd.rename('fwd')], axis=1).dropna()
    if len(df) < 50:
        continue
    ic_rows.append((D, spearmanr(df['sue'], df['fwd'])[0]))
    g = pd.qcut(df['sue'].rank(method='first'), 10, labels=False, duplicates='drop')
    group_rows.append(df.groupby(g)['fwd'].mean())

# ---- 8. 汇总 ----
ic = pd.Series(dict(ic_rows)).dropna()
print('\n===== SUE1 结果 =====')
print('月数:', len(ic))
print('RankIC 均值: %.4f' % ic.mean())
print('RankIC T值: %.2f' % (ic.mean() / ic.std() * np.sqrt(len(ic))))
print('月胜率: %.2f%%' % ((ic > 0).mean() * 100))

gret = pd.DataFrame(group_rows).mean()   # 各组月均收益（0=最低SUE ... 9=最高SUE）
print('\n各组月均收益(%), 0=不达预期 ... 9=超预期:')
print((gret * 100).round(2))
print('\n多空(9-0)月均收益: %.3f%%' % ((gret.iloc[9] - gret.iloc[0]) * 100))
print('多空约年化: %.2f%%' % ((gret.iloc[9] - gret.iloc[0]) * 12 * 100))
