import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- THE TAO ENGINE ---
def calculate_tao_mechanics(df):
    # 1. The EMA Stack (The Tao Core)
    for p in [8, 21, 34, 55, 89]:
        df[f'EMA{p}'] = df['Close'].ewm(span=p, adjust=False).mean()
    
    # 2. The 200 SMA (The Wind)
    df['SMA200'] = df['Close'].rolling(window=200).mean()
    
    # 3. ADX (Trend Strength) - Manual Calculation for Chromebook Compatibility
    n = 14
    h = df['High'].values.flatten()
    l = df['Low'].values.flatten()
    c = df['Close'].values.flatten()
    
    # Vectorized DM calculation
    plus_dm = np.insert(np.where((h[1:] - h[:-1]) > (l[:-1] - l[1:]), np.maximum(h[1:] - h[:-1], 0), 0), 0, 0)
    minus_dm = np.insert(np.where((l[:-1] - l[1:]) > (h[1:] - h[:-1]), np.maximum(l[:-1] - l[1:], 0), 0), 0, 0)
    tr = np.insert(np.maximum(h[1:] - l[1:], np.maximum(abs(h[1:] - c[:-1]), abs(l[1:] - c[:-1]))), 0, 0)
    
    # Ensure Series are 1D to prevent dimension errors
    atr_series = pd.Series(tr, index=df.index).rolling(window=n).mean()
    plus_di = 100 * (pd.Series(plus_dm, index=df.index).rolling(window=n).mean() / atr_series)
    minus_di = 100 * (pd.Series(minus_dm, index=df.index).rolling(window=n).mean() / atr_series)
    
    # Handle division by zero/NaN for DX calculation
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di)).fillna(0)
    
    df['ADX'] = dx.rolling(window=n).mean()
    df['ATR'] = atr_series
    return df

# --- UI SETUP ---
st.set_page_config(page_title="Tao Terminal", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for that "Sober Terminal" Look
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1f2937; padding: 15px; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

st.sidebar.title("🥷 Tao of Trading")
# Removed default stock for a clean start
ticker = st.sidebar.text_input("Enter Ticker:").upper()

if ticker:
    # Use 2y to ensure 200 SMA has enough data to stabilize
    data = yf.download(ticker, period="2y", interval="1d")
    
    if not data.empty and len(data) > 200:
        # Flatten MultiIndex columns if present (common with newer yfinance)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data = calculate_tao_mechanics(data)
        
        # Pull latest scalars correctly using .item() or float() to avoid Series ambiguity
        last_row = data.iloc[-1]
        price = float(last_row['Close'])
        sma200 = float(last_row['SMA200'])
        adx_val = float(last_row['ADX']) if not pd.isna(last_row['ADX']) else 0.0
        ema21 = float(last_row['EMA21'])
        atr = float(last_row['ATR']) if not pd.isna(last_row['ATR']) else 0.0
        
        # --- HEADER SECTION ---
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Current Price", f"${price:.2f}")
        col_m2.metric("ADX Strength", f"{adx_val:.1f}")
        col_m3.metric("ATR (Volatility)", f"${atr:.2f}")

        st.divider()

        # --- MAIN CONTENT ---
        chart_col, audit_col = st.columns([2, 1])

        with chart_col:
            st.subheader(f"📊 {ticker} Visual Audit")
            fig = go.Figure(data=[go.Candlestick(
                x=data.index, open=data['Open'], high=data['High'], 
                low=data['Low'], close=data['Close'], name="Price")])
            
            # Add the EMA Stack to the chart
            colors = ['#00ffcc', '#00ccff', '#3366ff', '#6633ff', '#ff33cc']
            for i, p in enumerate([8, 21, 34, 55, 89]):
                fig.add_trace(go.Scatter(x=data.index, y=data[f'EMA{p}'], 
                                         name=f'EMA{p}', line=dict(color=colors[i], width=1.5)))
            
            # Add the Wind (200 SMA)
            fig.add_trace(go.Scatter(x=data.index, y=data['SMA200'], 
                                     name='SMA 200 (The Wind)', line=dict(color='white', width=3, dash='dash')))
            
            fig.update_layout(template="plotly_dark", height=600, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig, use_container_width=True)

        with audit_col:
            st.subheader("⚙️ Mechanics Check")
            
            # 1. Trend (The Wind)
            if price > sma200:
                st.success("✅ SAILING WITH THE WIND: Price is above 200 SMA.")
            else:
                st.error("❌ STAGNANT WATER: Price is below 200 SMA. No Long setup.")

            # 2. The Stack - FIXED: Individual comparisons to avoid "truth value of Series" error
            e8, e21, e34, e55, e89 = (float(last_row['EMA8']), float(last_row['EMA21']), 
                                      float(last_row['EMA34']), float(last_row['EMA55']), float(last_row['EMA89']))
            
            is_stacked = (e8 > e21) and (e21 > e34) and (e34 > e55) and (e55 > e89)
            
            if is_stacked:
                st.success("✅ BULLISH STACK: EMAs are in perfect alignment.")
            else:
                st.warning("⚠️ DISORDERED STACK: Trend lacks momentum.")
            
            # Display Raw EMA List
            with st.expander("View Raw EMA Stack Data"):
                for p in [8, 21, 34, 55, 89]:
                    st.write(f"**EMA {p}:** ${float(last_row[f'EMA{p}']):.2f}")

            # 3. The Buy Zone (ATR logic)
            dist_to_21 = abs(price - ema21)
            in_buy_zone = dist_to_21 <= atr
            
            if in_buy_zone:
                st.info("🎯 IN THE BUY ZONE: Price is within 1 ATR of the 21 EMA.")
            else:
                st.warning("⌛ OVEREXTENDED: Price is too far from the mean. Wait for pullback.")

            # --- THE FINAL VERDICT ---
            st.divider()
            # Added ADX check to the final verdict logic
            if price > sma200 and is_stacked and adx_val >= 20 and in_buy_zone:
                st.balloons()
                st.markdown("### 🏆 HIGH QUALITY SETUP")
                st.write("All Tao mechanics are aligned for an entry.")
            else:
                st.markdown("### 🔍 MONITORING MODE")
                st.write("Wait for all mechanical criteria to align.")

    elif not data.empty:
        st.warning(f"Insufficient historical data for {ticker} (need at least 200 days).")
else:
    st.info("Enter a ticker in the sidebar to begin the Tao Audit.")
