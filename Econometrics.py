import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller

# ==============================================================================
# 1. PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(page_title="Quant Research Lab", layout="wide")
# --- LEGAL DISCLAIMER ---
with st.expander("⚖️ LEGAL DISCLAIMER / ZASTRZEŻENIE PRAWNE"):
    st.caption("""
    **ENG:** This application is for educational purposes only and presents quantitative financial models 
    based on historical data. It does not constitute investment advice or a recommendation to buy 
    or sell any financial instruments within the meaning of the Regulation of the Minister of Finance 
    or any other applicable law.
    
    **PL:** Program ma charakter wyłącznie edukacyjny i nie stanowi rekomendacji inwestycyjnej 
    w rozumieniu przepisów polskiego prawa. Inwestowanie wiąże się z ryzykiem utraty kapitału.
    """)

st.divider()
st.title("📉 Quant Research Lab: Market Efficiency Audit")

# --- METHODOLOGY INTRODUCTION ---
st.markdown("""
### 🔬 EMH Framework & Statistical Methodology. Author: Eryk Syldatk Certified Investment Analyst
Before running the audit, it is crucial to understand the environment you are analyzing.
The **Efficient Market Hypothesis (EMH)** suggests that asset prices fully reflect all available information. We specifically test for the **Weak Form of Efficiency**, investigating whether historical price patterns contain "memory" that can be exploited.
This audit determines whether it is even worth applying Technical Analysis to this asset.
**The goal is simple:** If the asset is "Efficient," any pattern you think you see is likely a statistical fluke. If it is "Inefficient," you have a **statistical green light** to explore further.
**Conclusion:** This audit doesn't tell you *when* to buy; it tells you *if* the asset is predictable enough to justify further Technical Analysis.
Here are the three pillars of our market efficiency testing framework:
""")

m_col1, m_col2, m_col3 = st.columns(3)

with m_col1:
    st.info("**1. Autocorrelation (Ljung-Box)**")
    st.caption("**Applied to: Log Returns**")
    st.write("Tests if past price movements influence future ones. If significant, the market has 'memory', contradicting the Random Walk hypothesis.")

with m_col2:
    st.info("**2. Stationarity (ADF Test)**")
    st.caption("**Applied to: Price Levels**")
    st.write("Checks for a unit root in the price series. While returns are usually stationary, prices should be non-stationary for the market to be efficient.")

with m_col3:
    st.info("**3. Normality (Jarque-Bera)**")
    st.caption("**Applied to: Log Returns**")
    st.write("Determines if returns follow a Normal Distribution. Significant results suggest 'Fat Tails' (excess kurtosis) and frequent market anomalies.")

st.divider()

# ==============================================================================
# 2. SIDEBAR - CONFIGURATION
# ==============================================================================
st.sidebar.header("🧪 Quant Research Lab")

# ASSET SELECTION
st.sidebar.subheader("1. Asset Selection")
st.sidebar.caption("⚠️ Use Yahoo Finance symbols (e.g., QQQ, BTC-USD, AAPL, GC=F)")
ticker = st.sidebar.text_input("Enter Ticker (Yahoo Finance)", value="ETH-USD")

# TIME PARAMETERS
st.sidebar.subheader("2. Time Horizon")
start_date = st.sidebar.date_input("Analysis Start Date", value=datetime.date(2021, 1, 1))
end_date = st.sidebar.date_input("End Date", value=datetime.date.today())

# ANALYSIS PARAMETERS
st.sidebar.subheader("3. Technical Parameters")

# Wybór interwału
interval_input = st.sidebar.selectbox(
    "Data Interval", 
    options=["Daily", "Weekly", "Monthly"], 
    index=0
)
# Mapowanie na format Yahoo Finance
interval_map = {"Daily": "1d", "Weekly": "1wk", "Monthly": "1mo"}

# Zmniejszony slider dla lagów
lags_input = st.sidebar.slider("Number of Lags", min_value=5, max_value=40, value=10)

st.sidebar.divider()
run_analysis = st.sidebar.button("🔍 RUN STATISTICAL AUDIT")

# ==============================================================================
# 3. DATA & CALCULATION FUNCTIONS
# ==============================================================================
# Update the function definition and the yf.download call
@st.cache_data(ttl=3600)
def fetch_data(symbol, start, end, interval): # Dodany parametr interval
    try:
        # Przekazujemy interval=interval do yf.download
        df = yf.download(symbol, start=start, end=end, interval=interval, progress=False)
        if df.empty: return None
        
        data = df[['Close']].copy()
        data['log_return'] = np.log(data['Close'] / data['Close'].shift(1))
        return data.dropna()
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# 1. AKCJA: Pobieranie danych (tylko po kliknięciu przycisku)
if run_analysis:
    with st.spinner(f"Analyzing {ticker}..."):
        fetched_df = fetch_data(ticker, start_date, end_date, interval_map[interval_input])
        if fetched_df is not None and not fetched_df.empty:
            st.session_state.df = fetched_df
            st.session_state.returns_series = fetched_df["log_return"].dropna().squeeze()
            st.session_state.prices_series = fetched_df["Close"].dropna().squeeze()
        else:
            st.error("❌ No data found.")
            st.stop()

# 2. WRAPPER (Śluza): Sprawdzamy, czy w ogóle mamy co analizować
if "df" in st.session_state and st.session_state.df is not None:
    # --- PRZYPISANIE I CZYSZCZENIE (isinstance zostaje tutaj!) ---
    df = st.session_state.df
    returns_series = st.session_state.returns_series
    prices_series = st.session_state.prices_series
    
    # Naprawa formatu (jeśli Series stało się DataFrame)
    if isinstance(returns_series, pd.DataFrame): 
        returns_series = returns_series.iloc[:, 0]
    if isinstance(prices_series, pd.DataFrame): 
        prices_series = prices_series.iloc[:, 0]
    
    n_obs = len(returns_series)
    
    # 3. STRAŻNIK ILOŚCI DANYCH
    if n_obs <= 10:
        st.error(f"⚠️ Too little data ($N = {n_obs}$)")
        st.stop()

    # --- NAGŁÓWEK AUDYTU ---
    st.header(f"🔍 Audit for {ticker}")
    if True:        
# --- SECTION 1: MARKET EFFICIENCY (LJUNG-BOX) ---
        st.divider()
        st.subheader("1. Market Efficiency & Autocorrelation")
        col1_1, col1_2 = st.columns([1, 2])
        
        n_obs = len(returns_series)
        lags_lb = min(20, n_obs // 2)
        
        # 1. Ljung-Box Test
        lb_df = acorr_ljungbox(returns_series, lags=[lags_lb], return_df=True)
        lb_p = lb_df['lb_pvalue'].values[0]
        
        # 2. ACF Calculation & Significance Threshold (The Blue Zone)
        acf_values = sm.tsa.acf(returns_series, nlags=lags_lb)
        conf_interval = 1.96 / np.sqrt(n_obs) # Threshold for alpha=0.05
        
        is_inefficient = lb_p < 0.05
        diag_mode = "None" 
        mode_desc = "🎲 Random Walk"

        if is_inefficient:
            # Logic: Search for the FIRST significant lag within the first 5 steps
            significant_lag = None
            for i in range(1, min(6, len(acf_values))):
                if np.abs(acf_values[i]) > conf_interval:
                    significant_lag = i
                    break
            
            # If a significant lag is found in 1-5, it dictates the mode.
            # Otherwise, we default to the sign of Lag 1.
            target_lag_idx = significant_lag if significant_lag else 1
            main_coef = acf_values[target_lag_idx]
            
            if main_coef > 0:
                diag_mode = "Momentum"
                mode_desc = f"📈 Momentum (Based on Significant Lag {target_lag_idx})"
            else:
                diag_mode = "Reversion"
                mode_desc = f"🪃 Mean Reversion (Based on Significant Lag {target_lag_idx})"
        
        # Save to session_state to ensure Section 5 works correctly
        st.session_state['diagnosis'] = diag_mode

        with col1_1:
            st.metric("Ljung-Box P-Value", f"{lb_p:.4f}")
            if is_inefficient:
                st.success("✅ **INEFFICIENT**")
                st.write(f"**Diagnosis:** {mode_desc}")
            else:
                st.error("❌ **EFFICIENT**")
                st.write(f"**Diagnosis:** {mode_desc}")

            with st.expander("📖 Universal Interpretation"):
                st.write(f"The blue confidence interval (95% CI) is currently: **±{conf_interval:.4f}**")
                st.write("- If a bar extends beyond this zone, the memory is statistically significant.")
                st.write("- **Positive Correlation:** Prices tend to persist (Trend/Momentum).")
                st.write("- **Negative Correlation:** Prices tend to bounce (Mean Reversion/Zig-zag).")

        with col1_2:
            fig_acf, ax_acf = plt.subplots(figsize=(10, 4))
            plot_lags = min(lags_input, n_obs - 1)
            sm.graphics.tsa.plot_acf(returns_series, lags=plot_lags, zero=False, ax=ax_acf, alpha=0.05)
            ax_acf.set_ylim(-0.2, 0.2)
            ax_acf.set_title(f"ACF Audit (Significance Threshold: ±{conf_interval:.3f})")
            ax_acf.grid(True, alpha=0.2)
            st.pyplot(fig_acf)

        # --- SECTION 2: STATIONARITY (ADF TEST) ---
        st.divider()
        st.subheader("2. Price Stationarity (Unit Root Test)")
        col2_1, col2_2 = st.columns([1, 2])
        
        uniques = prices_series.nunique()
        if uniques > 1 and len(prices_series) > 10:
            adf_res = adfuller(prices_series)
            adf_p = adf_res[1]
            is_stationary = adf_p < 0.05
            is_stationary_logic = is_stationary # To naprawi Twój NameError
        else:
            adf_p = 1.0
            is_stationary = False

        with col2_1:
            st.metric("ADF P-Value", f"{adf_p:.4f}")
            if is_stationary:
                st.success("✅ **STATIONARY**")
            else:
                st.error("❌ **NON-STATIONARY**")
            
            with st.expander("📖 Universal Interpretation"):
                st.write("""
                **Will it return to the mean?**
                - **Stationary (p < 0.05):** The price moves within a stable range or returns to its average. Best for Mean Reversion.
                - **Non-Stationary (p > 0.05):** The price has a trend or 'unit root'. It can go to 'infinity' without looking back.
                """)

        with col2_2:
            st.line_chart(df['Close'], use_container_width=True)

        # --- SECTION 3: NORMALITY & ANOMALIES ---
        st.divider()
        st.subheader("3. Returns Distribution & Normality")
        col3_1, col3_2 = st.columns([1, 2])
        
        jb_stat, jb_p = stats.jarque_bera(returns_series)

        with col3_1:
            st.metric("Jarque-Bera P-Value", f"{jb_p:.4f}")
            if jb_p < 0.05:
                st.warning("🧐 **NON-NORMAL**")
            else:
                st.success("✅ **NORMAL**")

            with st.expander("📖 Universal Interpretation"):
                st.write("""
                **Are 'Black Swans' likely?**
                - **Non-Normal (p < 0.05):** High 'Fat Tails'. Extreme crashes or pumps happen more often than a bell curve suggests.
                - **Normal (p > 0.05):** Returns are 'well-behaved'. Risk models (like VaR) are more reliable.
                """)

        with col3_2:
            fig_dist, ax_dist = plt.subplots(figsize=(10, 4))
            ax_dist.hist(returns_series, bins=60, color='#00FFAA', edgecolor='black', alpha=0.7)
            st.pyplot(fig_dist)

        # ==============================================================================
        # 4.5 FINAL QUANTITATIVE VERDICT (The Bridge)
        # ==============================================================================
        st.divider()
        st.header("🎯 Final Statistical Diagnosis")
        
        # Pobieramy współczynnik autokorelacji dla 1. laga (powinien być policzony w sekcji 4)
        # Jeśli nie, liczymy go tutaj:
        acf_vals = sm.tsa.acf(returns_series, nlags=1)
        lag_1_coef = acf_vals[1]

        # Logika werdyktu - szukamy pola do badań (Edge)
        has_edge = False
        edge_type = "None"

        # Diagnostic Logic
        if is_inefficient:
            if lag_1_coef > 0:
                st.success("### 💎 VERDICT: STRATEGIC EDGE DETECTED (Momentum)")
                st.write("""
                **Diagnosis:** Positive Autocorrelation found. The market exhibits 'Persistence.' 
                
                **👉 Next Steps for Technical Analysis:** Explore **Trend-Following** indicators.
                👉 Look for **Breakout Patterns** as moves tend to continue.
                """)
                diagnosis = "Momentum"
            else:
                st.info("### 🪃 VERDICT: STRATEGIC EDGE DETECTED (Mean Reversion)")
                st.write("""
                **Diagnosis:** Negative Autocorrelation found. The market exhibits 'Anti-Persistence' (Zig-Zag). 
                
                **👉 Next Steps for Technical Analysis:** Explore **Contrarian** setups. Fade the moves.
                👉Look for overextensions.
                """)
                diagnosis = "Reversion"

        elif is_stationary_logic:
            st.info("### 📉 VERDICT: STRATEGIC EDGE DETECTED (Stationary Range)")
            st.write("""
            **Diagnosis:** The price is 'Caged.' It lacks a long-term trend but has a strong gravitational pull to its mean.
            
            **👉 Next Steps for Technical Analysis:** Use **Range-Bound** tools.
            👉  Avoid trend-following indicators; they will likely give false signals here.
            """)
            diagnosis = "Range"

        else:
            st.error("### 🧱 VERDICT: NO STATISTICAL EDGE (Random Walk)")
            st.write("""
            **Diagnosis:** This asset is currently **Efficient**. 
            
            **⚠️ Warning:** Technical Analysis is highly unreliable here. Any patterns observed are likely 'Apophenia' (seeing patterns in random data). 
            - **Recommendation:** Save your capital. Wait for a regime change or move to a less efficient asset.
            """)

            diagnosis = "None"
    else:
    # Jeśli nikt jeszcze nie kliknął Execute
    st.info("👈 Set the parameters and click 'Execute Analysis' to start.")
    st.stop()




