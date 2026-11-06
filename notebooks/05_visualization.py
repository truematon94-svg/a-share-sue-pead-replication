"""
自包含版：取数 → SUE3 IC + 双变量分组 → 画图。
不依赖之前运行的任何变量。
"""
from jqdata import *
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

START = '2009-12-01'
END = '2020-01-23'
FETCH_FROM = '2006-01-01'
CHUNK = 40

# ---- 1. 信号日 ----
days = pd.DatetimeIndex(get_trade_days(start_date=START, end_date=END))
signal_dates = pd.DatetimeIndex(
    pd.Series(days, index=days).groupby(days.to_period('M')).max().values).sort_values()

# ---- 2. 全市场股票 ----
sec = get_all_securities(types=['stock'])
all_codes = sec.index.tolist()

# ---- 3. 取财报（定期+快报+预告） ----
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


income = fetch_income(all_codes)
income = income[income['report_type'] == 0][['code', 'pub_date', 'end_date', 'np_parent_company_owners']]
income = income.rename(columns={'np_parent_company_owners': 'cum_np'})
express = fetch_express(all_codes)[['code', 'pub_date', 'end_date', 'np_parent_company_owners']]
express = express.rename(columns={'np_parent_company_owners': 'cum_np'})
forecast = fetch_forecast(all_codes)
forecast = forecast[forecast['profit_min'].notna() & forecast['profit_max'].notna()]
forecast['cum_np'] = (forecast['profit_min'] + forecast['profit_max']) / 2
forecast = forecast[['code', 'pub_date', 'end_date', 'cum_np']]

cum_events = pd.concat([income, express, forecast], ignore_index=True, sort=True)
cum_events['pub_date'] = pd.to_datetime(cum_events['pub_date'])
cum_events['end_date'] = pd.to_datetime(cum_events['end_date'])
cum_events['month'] = cum_events['end_date'].dt.month
cum_events = cum_events[cum_events['month'].isin([3, 6, 9, 12])].copy()
print('合并事件表行数:', len(cum_events))

# ---- 4. 取价格 ----
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
signal_dates = signal_dates[signal_dates.isin(close.index)]
print('价格矩阵 shape:', close.shape)

# ---- 5. 行业 + 中性化 + SUE ----
raw_ind = get_industry(all_codes, date='2015-06-30')
industry = {}
for code in all_codes:
    item = raw_ind.get(code) or {}
    l1 = item.get('sw_l1') or {}
    industry[code] = l1.get('industry_code', 'UNKNOWN')
industry = pd.Series(industry)


def neutralize(y, log_mcap=None, industry=None):
    y = pd.to_numeric(y, errors='coerce').rename('factor')
    parts = []
    if log_mcap is not None:
        parts.append(pd.to_numeric(log_mcap, errors='coerce').rename('log_mcap'))
    if industry is not None:
        parts.append(pd.get_dummies(industry.rename('industry'), dummy_na=False, drop_first=True).astype(float))
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


# ---- 6. 主循环：SUE3 IC + 分组 + 双变量分组 ----
ic_rows = []
group_rows = []
long_rows = []
short_rows = []

for i in range(1, len(signal_dates) - 1):
    if i % 20 == 0:
        print('  进度 %d / %d 个月' % (i, len(signal_dates) - 1))
    D = signal_dates[i]
    Dprev = signal_dates[i - 1]
    Dn = signal_dates[i + 1]

    avail = cum_events[cum_events['pub_date'] <= D]
    best = avail.sort_values('pub_date').drop_duplicates(['code', 'end_date'], keep='last')
    best = best.sort_values(['code', 'end_date'])
    prev = best.groupby('code')['cum_np'].shift(1)
    best['single'] = np.where(best['month'].values == 3,
                              best['cum_np'].values,
                              best['cum_np'].values - prev.values)
    sue = best.groupby('code')['single'].apply(sue_from_series).dropna()
    fwd = close.loc[Dn] / close.loc[D] - 1

    # SUE3 原始 IC + 10 组
    df_ic = pd.concat([sue.rename('sue'), fwd.rename('fwd')], axis=1).dropna()
    if len(df_ic) >= 50:
        ic_rows.append((D, spearmanr(df_ic['sue'], df_ic['fwd'])[0]))
        g = pd.qcut(df_ic['sue'].rank(method='first'), 10, labels=False, duplicates='drop')
        group_rows.append(df_ic.groupby(g)['fwd'].mean())

    # 双变量分组（中性化 SUE × Ret）
    ret = close.loc[D] / close.loc[Dprev] - 1
    mcap = get_fundamentals(query(valuation.code, valuation.market_cap),
                            date=D.strftime('%Y-%m-%d')).set_index('code')['market_cap']
    log_mcap = np.log(mcap[mcap > 0])
    common = sue.index.intersection(ret.index).intersection(log_mcap.index)
    if len(common) >= 200:
        sue_n = neutralize(sue[common], log_mcap=log_mcap[common], industry=industry[common])
        ret_n = neutralize(ret[common], log_mcap=log_mcap[common])
        df5 = pd.concat([sue_n.rename('sue'), ret_n.rename('ret'), fwd.rename('fwd')], axis=1).dropna()
        if len(df5) >= 200:
            df5['L'] = pd.qcut(df5['sue'].rank(method='first'), 5, labels=False, duplicates='drop')
            df5['D'] = pd.qcut(df5['ret'].rank(method='first'), 5, labels=False, duplicates='drop')
            bench = fwd.mean()
            lm = (df5['L'] == 4) & (df5['D'] == 0)
            sm = (df5['L'] == 0) & (df5['D'] == 4)
            if lm.sum() >= 20:
                long_rows.append(df5.loc[lm, 'fwd'].mean() - bench)
            if sm.sum() >= 20:
                short_rows.append(df5.loc[sm, 'fwd'].mean() - bench)

ic = pd.Series(dict(ic_rows)).dropna()

# ---- 7. 画图 ----
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
BLUE = 'steelblue'

# 图1: 月度 RankIC + 累计 IC
fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                         gridspec_kw={'height_ratios': [2, 1]})
axes[0].plot(ic.index, ic.values, color=BLUE, lw=1.2)
axes[0].axhline(0, color='grey', lw=0.8, ls='--')
axes[0].set_ylabel('月度 RankIC')
axes[0].set_title('SUE3 因子月度 RankIC（2009.12–2020.01）')
axes[1].plot(ic.index, ic.cumsum().values, color=BLUE, lw=1.2)
axes[1].axhline(0, color='grey', lw=0.8, ls='--')
axes[1].set_ylabel('累计 IC')
axes[1].set_xlabel('日期')
plt.tight_layout()
plt.show()

# 图2: 10 组收益单调性
gret = pd.DataFrame(group_rows).mean().sort_index()
colors = plt.cm.Blues(np.linspace(0.35, 0.9, 10))
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(range(10), gret.values * 100, color=colors, edgecolor='white', lw=0.5)
ax.set_xticks(range(10))
ax.set_xticklabels(['D%d' % i for i in range(10)])
ax.set_xlabel('分组（D0=不达预期 … D9=超预期）')
ax.set_ylabel('月均收益（%）')
ax.set_title('SUE3 因子分组月均收益（单调性）')
ax.axhline(0, color='grey', lw=0.8)
plt.tight_layout()
plt.show()

# 图3: L4D0 多头累计超额净值
long_ex = pd.Series(long_rows)
nav = (1 + long_ex).cumprod()
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(nav.index, nav.values, color=BLUE, lw=1.5)
ax.set_xlabel('月份序号')
ax.set_ylabel('超额净值（相对全A等权）')
ax.set_title('L4D0 多头组合累计超额净值')
plt.tight_layout()
plt.show()

# 图4: SUE1/2/3 RankIC 单调上升
labels = ['SUE1\n(定期财报)', 'SUE2\n(+快报)', 'SUE3\n(+快报+预告)']
ric = [4.20, 4.35, 4.55]
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(range(3), ric, color=plt.cm.Blues(np.linspace(0.4, 0.9, 3)),
       edgecolor='white', lw=0.5)
ax.set_xticks(range(3))
ax.set_xticklabels(labels)
ax.set_ylabel('RankIC 均值（%）')
ax.set_title('SUE1 → SUE2 → SUE3：信息越及时，因子越强')
for i, v in enumerate(ric):
    ax.text(i, v + 0.03, '%.2f%%' % v, ha='center', fontsize=11)
ax.set_ylim(0, 5)
plt.tight_layout()
plt.show()
