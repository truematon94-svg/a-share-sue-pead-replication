"""
业绩超预期(SUE3) × 反转(Ret) 双变量分组。
取数 + 中性化 + 双变量分组 全在一段，不依赖之前运行状态。

目标：多头 L4D0(超预期+超跌) ≈ +11.78% 年化超额；空头 L0D4(不达预期+涨高) ≈ -19.12%。

待定/假设（研报未写明）：Ret 回看窗口=前1个月；行业=sw_l1取一次；市值=总市值取对数；基准=全A等权。
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

# ---- 3. 定期财报（利润表） ----
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
    return pd.concat(frames, ignore_index=True, sort=True)


income = fetch_income(all_codes)
income = income[income['report_type'] == 0].copy()
income = income.rename(columns={'np_parent_company_owners': 'cum_np'})
income = income[['code', 'pub_date', 'end_date', 'cum_np']]
print('定期财报行数(report_type=0):', len(income))

# ---- 4. 业绩快报 ----
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
    return pd.concat(frames, ignore_index=True, sort=True)


express = fetch_express(all_codes)
express = express.rename(columns={'np_parent_company_owners': 'cum_np'})
express = express[['code', 'pub_date', 'end_date', 'cum_np']]
print('业绩快报行数:', len(express))

# ---- 5. 业绩预告 ----
def fetch_forecast(codes):
    frames = []
    for i in range(0, len(codes), CHUNK):
        chunk = codes[i:i + CHUNK]
        q = query(finance.STK_FIN_FORCAST.code,
                  finance.STK_FIN_FORCAST.pub_date,
                  finance.STK_FIN_FORCAST.end_date,
                  finance.STK_FIN_FORCAST.profit_min,
                  finance.STK_FIN_FORCAST.profit_max).filter(
            finance.STK_FIN_FORCAST.pub_date >= FETCH_FROM,
            finance.STK_FIN_FORCAST.pub_date <= END,
            finance.STK_FIN_FORCAST.code.in_(chunk))
        frames.append(finance.run_query(q))
        if (i // CHUNK) % 20 == 0:
            print('  取业绩预告 %d/%d' % (i + len(chunk), len(codes)))
    return pd.concat(frames, ignore_index=True, sort=True)


forecast = fetch_forecast(all_codes)
forecast = forecast[forecast['profit_min'].notna() & forecast['profit_max'].notna()].copy()
forecast['cum_np'] = (forecast['profit_min'] + forecast['profit_max']) / 2
forecast = forecast[['code', 'pub_date', 'end_date', 'cum_np']]
print('业绩预告行数(有上下限):', len(forecast))

# ---- 6. 合并三类财报为"累计归母净利润事件表" ----
cum_events = pd.concat([income, express, forecast], ignore_index=True, sort=True)
cum_events['pub_date'] = pd.to_datetime(cum_events['pub_date'])
cum_events['end_date'] = pd.to_datetime(cum_events['end_date'])
cum_events['month'] = cum_events['end_date'].dt.month
cum_events = cum_events[cum_events['month'].isin([3, 6, 9, 12])].copy()
print('合并事件表行数:', len(cum_events))

# ---- 7. 价格 ----
def fetch_close(codes):
    parts = []
    for i in range(0, len(codes), 500):
        chunk = codes[i:i + 500]
        p = get_price(chunk, start_date=START, end_date=END,
                      frequency='daily', fields=['close'], fq='pre', skip_paused=False)
        if isinstance(p, pd.Panel):
            p = p['close']
        else:
            p = p[['close']].rename(columns={'close': chunk[0]})
        parts.append(p)
        if (i // 500) % 5 == 0:
            print('  取价格 %d/%d' % (i + len(chunk), len(codes)))
    return pd.concat(parts, axis=1, sort=True)


close = fetch_close(all_codes)
print('价格矩阵 shape:', close.shape)
signal_dates = signal_dates[signal_dates.isin(close.index)]


# ---- 8. 中性化函数（截面回归取残差） ----
def neutralize(y, log_mcap=None, industry=None):
    y = pd.to_numeric(y, errors='coerce').rename('factor')
    parts = []
    if log_mcap is not None:
        parts.append(pd.to_numeric(log_mcap, errors='coerce').rename('log_mcap'))
    if industry is not None:
        dummies = pd.get_dummies(industry.rename('industry'), dummy_na=False, drop_first=True)
        parts.append(dummies.astype(float))
    if not parts:
        return y - y.mean()
    X = pd.concat(parts, axis=1)
    data = pd.concat([y, X], axis=1).dropna()
    if len(data) <= X.shape[1] + 2:
        return pd.Series(index=y.index, dtype=float)
    yv = data['factor'].values.astype(float)
    xv = data.drop(columns='factor').values.astype(float)
    xv = np.column_stack([np.ones(len(xv)), xv])
    beta = np.linalg.lstsq(xv, yv, rcond=None)[0]
    resid = yv - xv @ beta
    out = pd.Series(np.nan, index=y.index, dtype=float)
    out.loc[data.index] = resid
    return out

# ---- 9. 行业（取一次，假设区间内不变） ----
raw_ind = get_industry(all_codes, date='2015-06-30')
industry = {}
for code in all_codes:
    item = raw_ind.get(code) or {}
    l1 = item.get('sw_l1') or {}
    industry[code] = l1.get('industry_code', 'UNKNOWN')
industry = pd.Series(industry, name='industry')
print('行业数(非UNKNOWN):', (industry != 'UNKNOWN').sum())

# ---- 10. 市值（逐月截面） ----
def fetch_mcap(date):
    q = query(valuation.code, valuation.market_cap)
    df = get_fundamentals(q, date=date)
    return df.set_index('code')['market_cap']

# ---- 11. SUE 计算 ----
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


# ---- 12. 逐月：SUE + Ret + 中性化 + 双变量分组 ----
long_rows = []
short_rows = []

for i in range(1, len(signal_dates) - 1):   # 从 1 开始：Ret 需要前一个信号日
    if i % 20 == 0:
        print('  进度 %d / %d 个月' % (i, len(signal_dates) - 1))
    D = signal_dates[i]
    Dprev = signal_dates[i - 1]
    Dn = signal_dates[i + 1]

    # (a) SUE（原始，点前）
    avail = cum_events[cum_events['pub_date'] <= D]
    best = avail.sort_values('pub_date').drop_duplicates(['code', 'end_date'], keep='last')
    best = best.sort_values(['code', 'end_date'])
    prev = best.groupby('code')['cum_np'].shift(1)
    best['single'] = np.where(best['month'].values == 3,
                              best['cum_np'].values,
                              best['cum_np'].values - prev.values)
    sue = best.groupby('code')['single'].apply(sue_from_series).dropna()

    # (b) Ret（前 1 个月收益）
    ret = close.loc[D] / close.loc[Dprev] - 1

    # (c) 市值
    mcap = fetch_mcap(D.strftime('%Y-%m-%d'))
    log_mcap = np.log(mcap[mcap > 0])

    # (d) 对齐 + 中性化（SUE 对行业+市值；Ret 只对市值）
    common = sue.index.intersection(ret.index).intersection(log_mcap.index)
    if len(common) < 200:
        continue
    sue_n = neutralize(sue[common], log_mcap=log_mcap[common], industry=industry[common])
    ret_n = neutralize(ret[common], log_mcap=log_mcap[common])

    # (e) 未来收益 + 全A等权基准
    fwd = close.loc[Dn] / close.loc[D] - 1
    bench = fwd.mean()

    df = pd.concat([sue_n.rename('sue'), ret_n.rename('ret'), fwd.rename('fwd')], axis=1).dropna()
    if len(df) < 200:
        continue
    df['L'] = pd.qcut(df['sue'].rank(method='first'), 5, labels=False, duplicates='drop')
    df['D'] = pd.qcut(df['ret'].rank(method='first'), 5, labels=False, duplicates='drop')

    long_mask = (df['L'] == 4) & (df['D'] == 0)   # SUE 最高层 且 Ret 最低组
    short_mask = (df['L'] == 0) & (df['D'] == 4)  # SUE 最低层 且 Ret 最高组
    if long_mask.sum() >= 20:
        long_rows.append(df.loc[long_mask, 'fwd'].mean() - bench)
    if short_mask.sum() >= 20:
        short_rows.append(df.loc[short_mask, 'fwd'].mean() - bench)

# ---- 13. 汇总 ----
long_ex = pd.Series(long_rows).mean()
short_ex = pd.Series(short_rows).mean()
print('\n===== 双变量分组 =====')
print('多头 L4D0 (超预期+超跌) 月均超额: %.3f%%' % (long_ex * 100))
print('多头 L4D0 约年化超额: %.2f%%' % (long_ex * 12 * 100))
print('空头 L0D4 (不达预期+涨高) 月均超额: %.3f%%' % (short_ex * 100))
print('空头 L0D4 约年化超额: %.2f%%' % (short_ex * 12 * 100))
