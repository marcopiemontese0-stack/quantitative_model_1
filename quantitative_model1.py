import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# prices = pd.read_csv("D:\\Altro\\Investimenti DEGIRO\\PORTFOLIO wD\\ptf_chevalier_python\\instrument_prices.csv")
url = "https://raw.githubusercontent.com/marcopiemontese0-stack/quantitative_model_1/main/instrument_prices.csv"
prices = pd.read_csv(url)

prices
ticker_list = prices.columns.tolist()

def quantitative_model(prices_df, tickers, momentum_weight, risk_adj_weight, risk_weight, verbose=True):
    results = {}
    
    for ticker in tickers:
        if verbose:
            print(f"\n" + "="*50)
            print(f"ANALISI TICKER: {ticker}")
        print("="*50)
        
        data = prices_df[['Date', ticker]].copy()
        data['Date'] = pd.to_datetime(data['Date'])
        data = data.sort_values('Date').reset_index(drop=True)
        data['returns'] = data[ticker].pct_change()

        # ==================== INDICATORI DI MOMENTUM ====================

        # Rendimenti cumulati su diversi orizzonti temporali
        data['mom_1m'] = data[ticker].pct_change(22)   # 1 mese
        data['mom_3m'] = data[ticker].pct_change(65)   # 3 mesi
        data['mom_6m'] = data[ticker].pct_change(130)  # 6 mesi
        data['mom_12m'] = data[ticker].pct_change(260) # 12 mesi

        # Momentum con skip (esclude il mese più recente per evitare reversal)
        data['mom_12_1'] = data[ticker].pct_change(260) - data[ticker].pct_change(22)

        # Medie mobili e crossover
        data['sma_50'] = data[ticker].rolling(50).mean()
        data['sma_200'] = data[ticker].rolling(200).mean()
        data['price_to_sma50'] = data[ticker] / data['sma_50'] - 1
        data['price_to_sma200'] = data[ticker] / data['sma_200'] - 1

        # Rate of Change (ROC)
        data['roc_10'] = (data[ticker] / data[ticker].shift(10) - 1) * 100
        data['roc_30'] = (data[ticker] / data[ticker].shift(30) - 1) * 100

        # Relative Strength Index (RSI)
        def calculate_rsi(series, period=14):
            delta = series.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi

        data['rsi_14'] = calculate_rsi(data[ticker], 14)

        # ==================== INDICATORI DI RISCHIO ====================

        # Volatilità (deviazione standard dei rendimenti)
        data['vol_1m'] = data['returns'].rolling(22).std() * np.sqrt(260)   # Annualizzata
        data['vol_3m'] = data['returns'].rolling(65).std() * np.sqrt(260)
        data['vol_1y'] = data['returns'].rolling(260).std() * np.sqrt(260)

        # Drawdown (massima perdita dal picco)
        data['cummax'] = data[ticker].cummax()
        data['drawdown'] = (data[ticker] / data['cummax'] - 1) * 100
        data['max_dd_1y'] = data['drawdown'].rolling(260).min()

        # Downside deviation (volatilità solo dei rendimenti negativi)
        def downside_deviation(returns, window):
            negative_returns = returns.where(returns < 0, 0)
            return negative_returns.rolling(window).std() * np.sqrt(260)

        data['downside_vol_3m'] = downside_deviation(data['returns'], 65)

        # Value at Risk (VaR) - 5° percentile
        data['var_5pct_3m'] = data['returns'].rolling(65).quantile(0.05)

        # Sharpe Ratio (assumendo risk-free rate = 0 per semplicità)
        data['sharpe_3m'] = (data['returns'].rolling(65).mean() * 260) / (data['returns'].rolling(65).std() * np.sqrt(260))
        data['sharpe_1y'] = (data['returns'].rolling(260).mean() * 260) / (data['returns'].rolling(260).std() * np.sqrt(260))

        # Sortino Ratio
        def sortino_ratio(returns, window):
            mean_return = returns.rolling(window).mean() * 260
            downside_std = downside_deviation(returns, window)
            return mean_return / downside_std

        data['sortino_3m'] = sortino_ratio(data['returns'], 65)

        # Calmar Ratio (rendimento annualizzato / max drawdown)
        data['calmar_1y'] = (data['returns'].rolling(260).mean() * 260) / abs(data['max_dd_1y'] / 100)

        # ==================== INDICATORI COMPOSITI ====================

        # Risk-Adjusted Momentum (momentum / volatilità)
        data['ram_3m'] = data['mom_3m'] / data['vol_3m']
        data['ram_6m'] = data['mom_6m'] / data['vol_3m']
        data['ram_12m'] = data['mom_12m'] / data['vol_1y']

        # Trend Strength
        data['trend_strength'] = abs(data['price_to_sma50']) * np.sign(data['mom_3m'])

        # Visualizza i risultati
        # if verbose: print(data.tail(10))

        # Seleziona gli indicatori chiave per il ranking
        momentum_features = ['mom_3m', 'mom_6m', 'mom_12_1', 'rsi_14']
        risk_features = ['vol_3m', 'max_dd_1y', 'sharpe_1y', 'sortino_3m']

        # Normalizzazione Z-score manuale (media 0, std 1)
        def normalize(series):
            return (series - series.mean()) / series.std()

        data_clean = data.dropna().copy()

        for feature in momentum_features + risk_features:
            data_clean[feature + '_norm'] = normalize(data_clean[feature])

        # Crea score composito
        data_clean['momentum_score'] = data_clean[['mom_3m_norm', 'mom_6m_norm', 'mom_12_1_norm']].mean(axis=1)
        data_clean.loc[data_clean['rsi_14'] > 70, 'momentum_score'] *= 0.8 # Se RSI > 70, diminuisce il momentum score (es. del 20%)
        data_clean.loc[data_clean['rsi_14'] < 30, 'momentum_score'] *= 1.2 # Se RSI < 30, aumenta il momentum score (es. del 20%)

        data_clean['risk_score'] = -data_clean[['vol_3m_norm']].mean(axis=1)
        data_clean['risk_adj_score'] = data_clean[['sharpe_1y_norm', 'sortino_3m_norm']].mean(axis=1)

        # Score finale
        data_clean['final_rank_score'] = (
            momentum_weight * data_clean['momentum_score'] + 
            risk_adj_weight * data_clean['risk_adj_score'] + 
            risk_weight * data_clean['risk_score']
        )

        # if verbose: print(data_clean[['Date', ticker, 'final_rank_score']].tail(20))
        
        data_plot = data_clean[['Date', ticker, 'final_rank_score']].copy()
        data_plot['Date'] = pd.to_datetime(data_plot['Date'])
        correlation = data_plot[ticker].corr(data_plot['final_rank_score'])
        high_score = data_plot[data_plot['final_rank_score'] > 1]
        low_score = data_plot[data_plot['final_rank_score'] < -1]

        if verbose:
            # ==================== GRAFICI ====================
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

            # ===== GRAFICO 1: Prezzo dell'asset =====
            ax1.plot(data_plot['Date'], data_plot[ticker], color="#101C22", linewidth=2, label='Prezzo')
            ax1.set_ylabel('Prezzo (EUR)', fontsize=12, fontweight='bold')
            ax1.set_title(f'{ticker} - Prezzo e Ranking Score', fontsize=14, fontweight='bold', pad=20)
            ax1.grid(True, alpha=0.3, linestyle='--')
            ax1.legend(loc='upper left', fontsize=10)

            # Colora lo sfondo in base al trend
            ax1.fill_between(data_plot['Date'], data_plot[ticker].min(), data_plot[ticker], 
                         alpha=0.1, color="#101C22")

            # ===== GRAFICO 2: Final Rank Score =====
            # Colora positivo/negativo con colori diversi
            ax2.fill_between(data_plot['Date'], 0, data_plot['final_rank_score'], 
                             where=(data_plot['final_rank_score'] >= 0), alpha=0.3, color='green', label='Score positivo')
            ax2.fill_between(data_plot['Date'], 0, data_plot['final_rank_score'], 
                            where=(data_plot['final_rank_score'] < 0), alpha=0.3, color='red', label='Score negativo')
            ax2.plot(data_plot['Date'], data_plot['final_rank_score'], color='black', linewidth=1.5, label='Final Rank Score')
            ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

            # Linee di riferimento per interpretazione
            ax2.axhline(y=1, color='green', linestyle=':', linewidth=1, alpha=0.5, label='Forte (±1 std)')
            ax2.axhline(y=-1, color='red', linestyle=':', linewidth=1, alpha=0.5)

            ax2.set_xlabel('Data', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Rank Score', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3, linestyle='--')
            ax2.legend(loc='upper left', fontsize=9)

            # Formattazione asse X
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.xticks(rotation=45)

            # Layout pulito
            plt.tight_layout()
            plt.show()

            # ===== STATISTICHE AGGIUNTIVE =====
            print("\n=== CORRELAZIONE PREZZO vs SCORE ===")
            print(f"Correlazione: {correlation:.3f}")

            print("\n=== PERIODI CON SCORE ESTREMO ===")
            print(f"Giorni con score > +1: {len(high_score)} ({len(high_score)/len(data_plot)*100:.1f}%)")
            print(f"Giorni con score < -1: {len(low_score)} ({len(low_score)/len(data_plot)*100:.1f}%)")

            print("\n=== PERFORMANCE MEDIA PER LIVELLO DI SCORE ===")
            # Dividi in quartili
            data_plot['score_quartile'] = pd.qcut(data_plot['final_rank_score'], q=4, labels=['Q1 (Basso)', 'Q2', 'Q3', 'Q4 (Alto)'])
            for q in sorted(data_plot['score_quartile'].unique(), reverse=True):
                subset = data_plot[data_plot['score_quartile'] == q]
                avg_price = subset[ticker].mean()
                print(f"{q}: Prezzo medio = {avg_price:.2f}")
        
        results[ticker] = data_clean
        statistics = statistics = {
            'correlation': round(correlation, 3),
            'days score > +1:': f"{len(high_score)/len(data_plot)*100:.1f}%",
            'days score < -1:': f"{len(low_score)/len(data_plot)*100:.1f}%"
        }

    return results, statistics


tickers = ['VWCEDE', 'IMAEAS', 'IS3NDE', 'AK8EDE', 'H4ZPDE', 'IB1TDE', 'IS3RDE', 'SXR8DE', 'SGLDMI', 'QDVEDE']

last_scores = {}
stats_df = {}
for ticker in tickers:
    result, stats = quantitative_model(prices, [ticker], 
                                       momentum_weight=0.4, 
                                       risk_adj_weight=0.3, 
                                       risk_weight=0.3, 
                                       verbose=False)
    last_scores[ticker] = result[ticker]['final_rank_score'].iloc[-1]
    stats_df[ticker] = stats  # correlation, high_score, low_score

scores_df = pd.DataFrame.from_dict(last_scores, orient='index', columns=['final_rank_score'])
scores_df = scores_df.sort_values('final_rank_score', ascending=False)
stats_df = pd.DataFrame.from_dict(stats_df, orient='index', columns=['correlation', 'days score > +1:', 'days score < -1:'])
final_df = pd.concat([scores_df, stats_df], axis=1)
final_df = final_df.sort_values('final_rank_score', ascending=False)
print(final_df)


# quantitative_model(prices, ['VWCEDE'], 0.5, 0.3, 0.2, verbose=True)['VWCEDE']['final_rank_score'].tail(1)
result, stats = quantitative_model(prices, ['IMAEAS'], 
                                   momentum_weight=0.4, 
                                   risk_adj_weight=0.3, 
                                   risk_weight=0.3, 
                                   verbose=True)


