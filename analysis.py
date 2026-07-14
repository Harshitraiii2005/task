import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

ORDER = ['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed']
COLORS = ['#8B0000', '#E74C3C', '#95A5A6', '#2ECC71', '#145A32']

# ---------------------------------------------------------------------------
# 1. Load & merge
# ---------------------------------------------------------------------------
trades = pd.read_csv('historical_data.csv')
fg = pd.read_csv('fear_greed_index.csv')

trades['date'] = pd.to_datetime(trades['Timestamp IST'], format='%d-%m-%Y %H:%M').dt.date
fg['date'] = pd.to_datetime(fg['date']).dt.date

df = trades.merge(fg[['date', 'classification', 'value']], on='date', how='left')
n_unmatched = df['classification'].isna().sum()
df = df.dropna(subset=['classification']).copy()

df['classification'] = pd.Categorical(df['classification'], categories=ORDER, ordered=True)
df['date'] = pd.to_datetime(df['date'])
df['net_pnl'] = df['Closed PnL'] - df['Fee']

closes = df[df['Closed PnL'] != 0].copy()
closes['return_pct'] = closes['Closed PnL'] / closes['Size USD'] * 100

print(f"Loaded {len(trades):,} trades, {n_unmatched} unmatched to sentiment "
      f"({n_unmatched / len(trades):.3%}).")
print(f"Merged dataset: {len(df):,} trades | {len(closes):,} closed (realized PnL) trades\n")

# ---------------------------------------------------------------------------
# 2. Summary table by sentiment
# ---------------------------------------------------------------------------
def summarize(g):
    c = g[g['Closed PnL'] != 0]
    return pd.Series({
        'n_trades': len(g),
        'n_closes': len(c),
        'total_pnl': g['Closed PnL'].sum(),
        'avg_pnl_per_trade': g['Closed PnL'].mean(),
        'avg_pnl_per_close': c['Closed PnL'].mean() if len(c) else np.nan,
        'win_rate_%': (c['Closed PnL'] > 0).mean() * 100 if len(c) else np.nan,
        'avg_size_usd': g['Size USD'].mean(),
        'median_size_usd': g['Size USD'].median(),
        'total_volume_usd': g['Size USD'].sum(),
        'buy_pct': (g['Side'] == 'BUY').mean() * 100,
    })

summary = df.groupby('classification', observed=True).apply(summarize).reindex(ORDER)
print("=== Performance summary by sentiment ===")
print(summary.round(2).to_string())
summary.to_csv('by_sentiment_summary.csv')

# ---------------------------------------------------------------------------
# 3. Extra stats printed to console (correlation, concentration, coins, sides)
# ---------------------------------------------------------------------------
daily2 = df.groupby('date').agg(daily_pnl=('Closed PnL', 'sum'), value=('value', 'mean')).reset_index()
print(f"\nCorrelation (FG numeric value vs daily total PnL): {daily2['daily_pnl'].corr(daily2['value']):.3f}")

acct_pnl = df.groupby('Account')['Closed PnL'].sum().sort_values(ascending=False)
top5_share = acct_pnl.head(5).sum() / acct_pnl.sum() * 100
print(f"Top-5-account share of total PnL: {top5_share:.1f}%  (of {df['Account'].nunique()} accounts)")

coin_pnl = df.groupby('Coin')['Closed PnL'].sum().sort_values(ascending=False)
print("\nTop 5 coins by PnL:\n", coin_pnl.head(5).round(0))
print("\nBottom 5 coins by PnL:\n", coin_pnl.tail(5).round(0))

side_perf = closes.groupby(['classification', 'Side'], observed=True)['Closed PnL'].mean().unstack()
print("\nAvg PnL per closed trade, Long vs Short, by sentiment:\n", side_perf.round(2))

# ---------------------------------------------------------------------------
# 4. Charts
# ---------------------------------------------------------------------------
plt.rcParams['font.size'] = 11

# Chart 1 — Total PnL by sentiment
tot = df.groupby('classification', observed=True)['Closed PnL'].sum().reindex(ORDER)
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(ORDER, tot.values / 1e6, color=COLORS)
ax.set_ylabel('Total Closed PnL ($ Millions)')
ax.set_title('Total Trader PnL by Market Sentiment (2023-2025)')
ax.bar_label(bars, fmt='%.2f')
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig('chart1_total_pnl.png', dpi=150)
plt.close()

# Chart 2 — Win rate by sentiment
wr = closes.groupby('classification', observed=True)['Closed PnL'].apply(lambda s: (s > 0).mean() * 100).reindex(ORDER)
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(ORDER, wr.values, color=COLORS)
ax.set_ylabel('Win Rate (%)')
ax.set_title('Win Rate of Closed Trades by Market Sentiment')
ax.bar_label(bars, fmt='%.1f%%')
ax.set_ylim(0, 100)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig('chart2_win_rate.png', dpi=150)
plt.close()

# Chart 3 — Median return % by sentiment (robust to outliers)
med_ret = closes.groupby('classification', observed=True)['return_pct'].median().reindex(ORDER)
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(ORDER, med_ret.values, color=COLORS)
ax.set_ylabel('Median Return per Closed Trade (%)')
ax.set_title('Median Trade Return by Market Sentiment')
ax.bar_label(bars, fmt='%.2f%%')
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig('chart3_median_return.png', dpi=150)
plt.close()

# Chart 4 — Trading activity (volume + count) by sentiment
vol = df.groupby('classification', observed=True)['Size USD'].sum().reindex(ORDER) / 1e6
cnt = df.groupby('classification', observed=True).size().reindex(ORDER)
x = np.arange(len(ORDER))
w = 0.35
fig, ax1 = plt.subplots(figsize=(9, 5))
ax1.bar(x - w / 2, vol.values, w, label='Total $ Volume (M)', color='#3498DB')
ax1.set_ylabel('Total Volume ($ Millions)', color='#3498DB')
ax2 = ax1.twinx()
ax2.bar(x + w / 2, cnt.values, w, label='Trade Count', color='#F39C12')
ax2.set_ylabel('Trade Count', color='#F39C12')
ax1.set_xticks(x)
ax1.set_xticklabels(ORDER, rotation=15)
ax1.set_title('Trading Activity by Market Sentiment')
plt.tight_layout()
plt.savefig('chart4_activity.png', dpi=150)
plt.close()

# Chart 5 — Cumulative PnL over time
daily_all = df.groupby('date', observed=True).agg(daily_pnl=('Closed PnL', 'sum')).reset_index()
fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(daily_all['date'], daily_all['daily_pnl'].cumsum(), color='#2C3E50', linewidth=1.8)
ax.set_ylabel('Cumulative Closed PnL ($)')
ax.set_title('Cumulative Trader PnL Over Time (2023-2025)')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('chart5_cumulative_pnl.png', dpi=150)
plt.close()

# Chart 6 — Median position size by sentiment
size_med = df.groupby('classification', observed=True)['Size USD'].median().reindex(ORDER)
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(ORDER, size_med.values, color=COLORS)
ax.set_ylabel('Median Trade Size (USD)')
ax.set_title('Median Position Size by Market Sentiment')
ax.bar_label(bars, fmt='$%.0f')
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig('chart6_position_size.png', dpi=150)
plt.close()

# Chart 7 — Long vs Short PnL by sentiment
side_pnl = closes.groupby(['classification', 'Side'], observed=True)['Closed PnL'].sum().unstack().reindex(ORDER)
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x - w / 2, side_pnl['BUY'] / 1000, w, label='Long (Buy)', color='#27AE60')
ax.bar(x + w / 2, side_pnl['SELL'] / 1000, w, label='Short (Sell)', color='#C0392B')
ax.set_xticks(x)
ax.set_xticklabels(ORDER, rotation=15)
ax.set_ylabel('Total Closed PnL ($ Thousands)')
ax.set_title('Long vs Short PnL by Market Sentiment')
ax.legend()
ax.axhline(0, color='black', linewidth=0.8)
plt.tight_layout()
plt.savefig('chart7_long_short.png', dpi=150)
plt.close()

print("\nAll 7 charts and by_sentiment_summary.csv written to the current directory.")