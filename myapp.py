import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# Set page configuration
st.set_page_config(
    page_title="Advanced Stock Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced custom CSS
st.markdown("""
    <style>
    .stAlert {
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 24px;
        background-color: #f0f2f6;
        border-radius: 5px;
    }
    .css-1d391kg {
        padding: 1rem 1rem 1.5rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 28px;
    }
    div[data-testid="stMetricDelta"] {
        font-size: 16px;
    }
    </style>
    """, unsafe_allow_html=True)

class StockAnalyzer:
    def __init__(self, symbol: str, lookback_years: int = 5):
        self.symbol = symbol.upper()
        self.lookback_years = lookback_years
        self.start_date = pd.Timestamp.now() - pd.DateOffset(years=lookback_years)
        
    def fetch_data(self):
        """Fetch stock data with error handling"""
        try:
            with st.spinner(f"Fetching data for {self.symbol}..."):
                ticker = yf.Ticker(self.symbol)
                df = ticker.history(period='1d', start=self.start_date)
                if df.empty:
                    raise ValueError(f"No data available for {self.symbol}")
                
                # Add company info
                self.info = ticker.info
                return df
        except Exception as e:
            raise ValueError(f"Error fetching data for {self.symbol}: {str(e)}")

    @staticmethod
    def calculate_volatility(prices, window=20):
        """Calculate rolling volatility"""
        returns = np.log(prices/prices.shift(1))
        return returns.rolling(window=window).std() * np.sqrt(252)

    @staticmethod
    def calculate_rsi(prices, period=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def calculate_macd(prices, fast=12, slow=26, signal=9):
        """Calculate MACD indicator"""
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd, signal_line

def main():
    # Title and description with company info container
    st.title("📊 Advanced Stock Analysis Dashboard")
    
    # Sidebar controls with enhanced UI
    with st.sidebar:
        st.header("📈 Analysis Parameters")
        
        # Add a search box with popular stocks
        popular_stocks = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META", "NVDA", "JPM"]
        symbol = st.selectbox(
            "Select or enter stock symbol",
            options=[""] + popular_stocks,
            index=0
        ).upper()
        
        if not symbol:
            symbol = st.text_input("Or enter any stock symbol", value="").upper()

        # Analysis parameters in an expander
        with st.expander("⚙️ Advanced Settings", expanded=True):
            lookback_years = st.slider("Lookback Period (Years)", 1, 15, 5)
            confidence_level = st.slider("Confidence Level (%)", 80, 99, 95)
            vol_window = st.slider("Volatility Window (Days)", 5, 50, 20)
            
        # Help section
        with st.expander("ℹ️ Help & Information"):
            st.markdown("""
            ### How to use this dashboard:
            1. Enter a stock symbol or select from popular stocks
            2. Adjust analysis parameters if needed
            3. Explore different tabs for detailed analysis
            
            ### Available Metrics:
            - Price trends and volume analysis
            - Volatility patterns
            - RSI (Relative Strength Index)
            - MACD (Moving Average Convergence Divergence)
            - Statistical measures and risk metrics
            """)

    if symbol:
        try:
            # Initialize analyzer and fetch data
            analyzer = StockAnalyzer(symbol, lookback_years)
            df = analyzer.fetch_data()
            
            # Display company info
            with st.expander("🏢 Company Information", expanded=True):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Market Cap", f"${analyzer.info.get('marketCap', 0)/1e9:.2f}B")
                with col2:
                    st.metric("Sector", analyzer.info.get('sector', 'N/A'))
                with col3:
                    st.metric("Industry", analyzer.info.get('industry', 'N/A'))
                
                st.markdown(f"**Description:** {analyzer.info.get('longBusinessSummary', 'N/A')}")
            
            # Calculate indicators
            df['Volatility'] = analyzer.calculate_volatility(df['Close'], window=vol_window)
            df['RSI'] = analyzer.calculate_rsi(df['Close'])
            df['MACD'], df['Signal'] = analyzer.calculate_macd(df['Close'])

            # Create tabs for different analysis sections
            tab1, tab2, tab3, tab4 = st.tabs([
                "📈 Price Analysis",
                "📊 Technical Indicators",
                "🎯 Risk Metrics",
                "📉 Performance Analysis"
            ])

            with tab1:
                st.subheader("Price Movement Analysis")
                
                # Time period selector
                period = st.radio(
                    "Select time period",
                    ["1M", "3M", "6M", "1Y", "All"],
                    horizontal=True
                )
                
                periods = {
                    "1M": 30, "3M": 90, "6M": 180,
                    "1Y": 252, "All": len(df)
                }
                df_period = df.tail(periods[period])
                
                # Enhanced price chart
                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=df_period.index,
                    open=df_period['Open'],
                    high=df_period['High'],
                    low=df_period['Low'],
                    close=df_period['Close'],
                    name='Price'
                ))
                
                fig.add_trace(go.Bar(
                    x=df_period.index,
                    y=df_period['Volume'],
                    name='Volume',
                    yaxis='y2',
                    opacity=0.3
                ))
                
                fig.update_layout(
                    yaxis2=dict(
                        title="Volume",
                        overlaying="y",
                        side="right"
                    ),
                    height=600,
                    title=f"{symbol} Price and Volume Chart",
                    yaxis_title="Price",
                    xaxis_title="Date",
                    template="plotly_white",
                    showlegend=True,
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)

                # Key metrics with enhanced styling
                st.markdown("### 📊 Key Metrics")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    current_price = df['Close'][-1]
                    prev_price = df['Close'][-2]
                    price_change = ((current_price/prev_price)-1)*100
                    st.metric(
                        "Current Price",
                        f"${current_price:.2f}",
                        f"{price_change:.2f}%",
                        delta_color="normal"
                    )
                with col2:
                    st.metric(
                        "52-Week High",
                        f"${df['High'][-252:].max():.2f}",
                        f"{((df['Close'][-1]/df['High'][-252:].max())-1)*100:.2f}%"
                    )
                with col3:
                    st.metric(
                        "52-Week Low",
                        f"${df['Low'][-252:].min():.2f}",
                        f"{((df['Close'][-1]/df['Low'][-252:].min())-1)*100:.2f}%"
                    )
                with col4:
                    avg_vol = df['Volume'].mean()
                    curr_vol = df['Volume'][-1]
                    vol_change = ((curr_vol/avg_vol)-1)*100
                    st.metric(
                        "Volume",
                        f"{curr_vol:,.0f}",
                        f"{vol_change:.2f}% vs avg"
                    )

            with tab2:
                st.subheader("Technical Indicators")
                
                # Technical indicators in columns
                col1, col2 = st.columns(2)
                
                with col1:
                    # RSI Plot
                    fig_rsi = go.Figure()
                    fig_rsi.add_trace(go.Scatter(
                        x=df_period.index,
                        y=df_period['RSI'],
                        name='RSI',
                        line=dict(color='blue')
                    ))
                    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
                    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
                    fig_rsi.update_layout(
                        title="RSI (Relative Strength Index)",
                        yaxis_title="RSI",
                        template="plotly_white"
                    )
                    st.plotly_chart(fig_rsi, use_container_width=True)
                    
                    current_rsi = df['RSI'][-1]
                    rsi_status = "Overbought" if current_rsi > 70 else "Oversold" if current_rsi < 30 else "Neutral"
                    st.metric("Current RSI", f"{current_rsi:.2f}", rsi_status)
                
                with col2:
                    # MACD Plot
                    fig_macd = go.Figure()
                    fig_macd.add_trace(go.Scatter(
                        x=df_period.index,
                        y=df_period['MACD'],
                        name='MACD',
                        line=dict(color='blue')
                    ))
                    fig_macd.add_trace(go.Scatter(
                        x=df_period.index,
                        y=df_period['Signal'],
                        name='Signal Line',
                        line=dict(color='orange')
                    ))
                    fig_macd.update_layout(
                        title="MACD (Moving Average Convergence Divergence)",
                        yaxis_title="MACD",
                        template="plotly_white"
                    )
                    st.plotly_chart(fig_macd, use_container_width=True)
                    
                    macd_value = df['MACD'][-1]
                    signal_value = df['Signal'][-1]
                    macd_status = "Bullish" if macd_value > signal_value else "Bearish"
                    st.metric("MACD Signal", f"{macd_value-signal_value:.4f}", macd_status)

            with tab3:
                st.subheader("Risk Analysis")
                
                # Volatility analysis
                fig_vol = go.Figure()
                fig_vol.add_trace(go.Scatter(
                    x=df_period.index,
                    y=df_period['Volatility']*100,
                    name='Volatility',
                    fill='tozeroy',
                    line=dict(color='purple')
                ))
                fig_vol.update_layout(
                    title="Historical Volatility",
                    yaxis_title="Volatility (%)",
                    template="plotly_white"
                )
                st.plotly_chart(fig_vol, use_container_width=True)
                
                # Risk metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    current_vol = df['Volatility'][-1]*100
                    avg_vol = df['Volatility'].mean()*100
                    st.metric(
                        "Current Volatility",
                        f"{current_vol:.2f}%",
                        f"{current_vol-avg_vol:.2f}% vs avg"
                    )
                with col2:
                    # Calculate Value at Risk
                    returns = df['Close'].pct_change()
                    var = np.percentile(returns, 100-confidence_level)
                    st.metric(
                        f"Value at Risk ({confidence_level}%)",
                        f"{-var*100:.2f}%"
                    )
                with col3:
                    # Calculate Sharpe Ratio
                    risk_free_rate = 0.02  # Assuming 2% risk-free rate
                    excess_returns = returns - risk_free_rate/252
                    sharpe = np.sqrt(252) * excess_returns.mean() / returns.std()
                    st.metric("Sharpe Ratio", f"{sharpe:.2f}")

            with tab4:
                st.subheader("Performance Analysis")
                
                # Calculate returns for different periods
                returns_periods = {
                    "1D": 1,
                    "1W": 5,
                    "1M": 21,
                    "3M": 63,
                    "6M": 126,
                    "1Y": 252
                }
                
                # Create returns table
                returns_data = []
                for period, days in returns_periods.items():
                    if len(df) >= days:
                        period_return = (df['Close'][-1] / df['Close'][-days] - 1) * 100
                        returns_data.append({
                            "Period": period,
                            "Return (%)": f"{period_return:.2f}%"
                        })
                
                returns_df = pd.DataFrame(returns_data)
                
                # Display returns in a clean table
                st.markdown("### 📊 Returns Analysis")
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.dataframe(
                        returns_df.style.format({"Return (%)": "{}"})
                            .set_properties(**{'text-align': 'center'}),
                        use_container_width=True
                    )
                
                with col2:
                    # Create returns visualization
                    fig_returns = go.Figure()
                    period_returns = [float(x["Return (%)"].strip('%')) for x in returns_data]
                    
                    fig_returns.add_trace(go.Bar(
                        x=[x["Period"] for x in returns_data],
                        y=period_returns,
                        marker_color=['red' if x < 0 else 'green' for x in period_returns]
                    ))
                    
                    fig_returns.update_layout(
                        title="Returns by Time Period",
                        xaxis_title="Time Period",
                        yaxis_title="Return (%)",
                        template="plotly_white",
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig_returns, use_container_width=True)
                
                # Additional performance metrics
                st.markdown("### 📈 Performance Metrics")
                col1, col2, col3, col4 = st.columns(4)
                
                # Calculate daily returns
                daily_returns = df['Close'].pct_change()
                
                with col1:
                    # Annualized Return
                    total_days = (df.index[-1] - df.index[0]).days
                    total_return = (df['Close'][-1] / df['Close'][0]) - 1
                    ann_return = (1 + total_return) ** (365/total_days) - 1
                    st.metric(
                        "Annualized Return",
                        f"{ann_return*100:.2f}%"
                    )
                
                with col2:
                    # Max Drawdown
                    rolling_max = df['Close'].cummax()
                    drawdowns = (df['Close'] - rolling_max) / rolling_max
                    max_drawdown = drawdowns.min()
                    st.metric(
                        "Maximum Drawdown",
                        f"{max_drawdown*100:.2f}%"
                    )
                
                with col3:
                    # Beta (using S&P 500 as benchmark)
                    try:
                        sp500 = yf.download('^GSPC', start=df.index[0], end=df.index[-1])['Close']
                        sp500_returns = sp500.pct_change()
                        beta = np.cov(daily_returns.dropna(), sp500_returns.dropna())[0,1] / np.var(sp500_returns.dropna())
                        st.metric(
                            "Beta (vs S&P 500)",
                            f"{beta:.2f}"
                        )
                    except:
                        st.metric("Beta (vs S&P 500)", "N/A")
                
                with col4:
                    # Risk-adjusted return (Sortino Ratio)
                    risk_free_rate = 0.02  # Assuming 2% risk-free rate
                    excess_returns = daily_returns - risk_free_rate/252
                    downside_returns = excess_returns[excess_returns < 0]
                    sortino = np.sqrt(252) * excess_returns.mean() / downside_returns.std()
                    st.metric(
                        "Sortino Ratio",
                        f"{sortino:.2f}"
                    )
                
                # Return Distribution Analysis
                st.markdown("### 📊 Return Distribution")
                fig_dist = go.Figure()
                
                # Create histogram of returns
                fig_dist.add_trace(go.Histogram(
                    x=daily_returns.dropna() * 100,
                    nbinsx=50,
                    name='Daily Returns',
                    showlegend=True
                ))
                
                # Add a normal distribution curve for comparison
                x_range = np.linspace(daily_returns.min() * 100, daily_returns.max() * 100, 100)
                fig_dist.add_trace(go.Scatter(
                    x=x_range,
                    y=stats.norm.pdf(x_range, daily_returns.mean() * 100, daily_returns.std() * 100) * len(daily_returns) * (daily_returns.max() - daily_returns.min()) * 100 / 50,
                    name='Normal Distribution',
                    line=dict(color='red', dash='dash')
                ))
                
                fig_dist.update_layout(
                    title="Daily Returns Distribution",
                    xaxis_title="Daily Return (%)",
                    yaxis_title="Frequency",
                    template="plotly_white",
                    showlegend=True
                )
                
                st.plotly_chart(fig_dist, use_container_width=True)
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")
            st.info("Please try refreshing the page or contact support if the issue persists.")

if __name__ == "__main__":
    main()        