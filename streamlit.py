# trading_web.py
"""
Web-Based Real-Time Trading Dashboard
Deploy to Streamlit Cloud, Hugging Face, or any web hosting
"""

import os
import json
import time
import random
import asyncio
import threading
import datetime
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Deque, Any
from collections import deque
from enum import Enum
from contextlib import contextmanager
import queue

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Optional: Only import web libraries if needed
try:
    import streamlit as st
    import plotly.graph_objects as go
    import plotly.subplots as make_subplots
    from plotly.subplots import make_subplots
    WEB_AVAILABLE = True
except ImportError:
    WEB_AVAILABLE = False
    print("Web libraries not installed. Install with: pip install streamlit plotly")

# Load environment variables
load_dotenv()

# ============================================================================
# Configuration
# ============================================================================

class StreamMode(Enum):
    MOCK = "mock"
    ALPACA = "alpaca"
    REPLAY = "replay"

@dataclass
class TradingConfig:
    """Configuration for trading system"""
    # Trading Parameters
    symbol: str = "SPY"
    short_window: int = 10
    long_window: int = 30
    max_data_points: int = 200
    buy_allocation: float = 0.5
    initial_cash: float = 10000.00
    
    # Risk Management
    stop_loss_pct: float = 0.03
    take_profit_pct: float = 0.08
    risk_per_trade: float = 0.02
    
    # System Settings
    stream_mode: StreamMode = StreamMode.MOCK
    update_interval_ms: int = 100  # Web update interval
    
    def __post_init__(self):
        """Validate configuration"""
        if self.stream_mode == StreamMode.ALPACA:
            if not os.getenv("ALPACA_API_KEY") or not os.getenv("ALPACA_SECRET_KEY"):
                print("⚠️  API keys missing, defaulting to MOCK mode")
                self.stream_mode = StreamMode.MOCK

# ============================================================================
# Thread-Safe Data Storage
# ============================================================================

class WebSafeDeque:
    """Thread-safe deque for web applications"""
    
    def __init__(self, maxlen=200):
        self._deque = deque(maxlen=maxlen)
        self._lock = threading.RLock()
        
    def append(self, item):
        with self._lock:
            self._deque.append(item)
            
    def extend(self, items):
        with self._lock:
            self._deque.extend(items)
            
    def __len__(self):
        with self._lock:
            return len(self._deque)
            
    def to_list(self):
        with self._lock:
            return list(self._deque)
            
    def last(self, n=1):
        with self._lock:
            if len(self._deque) == 0:
                return [] if n > 1 else None
            return list(self._deque)[-n:]

class TradingState:
    """Thread-safe state for web applications"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.lock = threading.RLock()
        
        # Data storage
        self.price_data = WebSafeDeque(config.max_data_points)
        self.sma_short_data = WebSafeDeque(config.max_data_points)
        self.sma_long_data = WebSafeDeque(config.max_data_points)
        self.timestamps = WebSafeDeque(config.max_data_points)
        
        # Trading signals
        self.signal_history = WebSafeDeque(config.max_data_points)
        self.transaction_history = WebSafeDeque(config.max_data_points)
        self.transaction_value_history = WebSafeDeque(config.max_data_points)
        
        # Portfolio
        self.portfolio = {
            "cash": config.initial_cash,
            "shares": 0.0,
            "history": WebSafeDeque(config.max_data_points),
            "total_trades": 0,
            "winning_trades": 0,
            "total_pnl": 0.0
        }
        
        # Current state
        self.entry_price = 0.0
        self.last_update = time.time()
        self.is_running = True
        
    def snapshot(self) -> Dict[str, Any]:
        """Get thread-safe snapshot of current state"""
        with self.lock:
            return {
                "prices": self.price_data.to_list(),
                "sma_short": self.sma_short_data.to_list(),
                "sma_long": self.sma_long_data.to_list(),
                "signals": self.signal_history.to_list(),
                "portfolio_value": self.portfolio["history"].to_list(),
                "current_cash": self.portfolio["cash"],
                "current_shares": self.portfolio["shares"],
                "total_trades": self.portfolio["total_trades"],
                "total_pnl": self.portfolio["total_pnl"],
                "entry_price": self.entry_price,
                "is_running": self.is_running,
                "last_update": self.last_update
            }
    
    def calculate_current_value(self, current_price: float = None) -> float:
        """Calculate current portfolio value"""
        with self.lock:
            if current_price is None:
                if len(self.price_data) > 0:
                    current_price = self.price_data.last()[0]
                else:
                    current_price = 0
            
            return self.portfolio["cash"] + (self.portfolio["shares"] * current_price)

# ============================================================================
# Trading Engine (Optimized for Web)
# ============================================================================

class WebTradingEngine:
    """Trading engine optimized for web applications"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.state = TradingState(config)
        self.data_queue = queue.Queue(maxsize=1000)
        self.is_running = False
        self.thread = None
        
    def start(self):
        """Start the trading engine"""
        if self.is_running:
            return
            
        self.is_running = True
        self.state.is_running = True
        
        # Start data generation in background thread
        self.thread = threading.Thread(target=self._run_engine, daemon=True)
        self.thread.start()
        
        print("🚀 Trading engine started")
        
    def stop(self):
        """Stop the trading engine"""
        self.is_running = False
        self.state.is_running = False
        if self.thread:
            self.thread.join(timeout=2)
        print("🛑 Trading engine stopped")
    
    def _run_engine(self):
        """Main engine loop running in background thread"""
        # Initialize price
        price = self.config.initial_cash / 10
        trend_direction = 1
        trend_counter = 0
        trend_duration = random.randint(50, 200)
        
        while self.is_running:
            try:
                # Generate mock data (replace with Alpaca if needed)
                trend_counter += 1
                if trend_counter >= trend_duration:
                    trend_direction *= -1
                    trend_duration = random.randint(50, 200)
                    trend_counter = 0
                
                # Simulate price movement
                drift = 0.0001
                volatility = random.gauss(0, 0.02)
                trend = trend_direction * 0.001
                
                price *= (1 + drift + trend + volatility)
                price = max(0.01, price)
                
                # Process the tick
                self._process_tick(price)
                
                # Sleep to control update rate
                time.sleep(0.1)  # 10 ticks per second
                
            except Exception as e:
                print(f"Engine error: {e}")
                time.sleep(1)
    
    def _process_tick(self, price: float):
        """Process a single price tick"""
        with self.state.lock:
            # Record price
            timestamp = datetime.datetime.now().isoformat()
            self.state.price_data.append(price)
            self.state.timestamps.append(timestamp)
            
            # Calculate SMAs
            sma_short = self._calculate_sma(self.state.price_data, self.config.short_window)
            sma_long = self._calculate_sma(self.state.price_data, self.config.long_window)
            
            self.state.sma_short_data.append(sma_short)
            self.state.sma_long_data.append(sma_long)
            
            # Generate trading signals
            signal = 0
            trade_amount = 0.0
            trade_value = 0.0
            
            if sma_short and sma_long:
                # Check for buy signal
                if sma_short > sma_long and self.state.portfolio["shares"] == 0:
                    # Execute buy
                    cash_available = self.state.portfolio["cash"]
                    spend_amount = cash_available * self.config.buy_allocation
                    shares_to_buy = spend_amount / price
                    
                    if shares_to_buy > 0:
                        self.state.portfolio["shares"] += shares_to_buy
                        self.state.portfolio["cash"] -= spend_amount
                        self.state.entry_price = price
                        self.state.portfolio["total_trades"] += 1
                        
                        signal = 1
                        trade_amount = shares_to_buy
                        trade_value = spend_amount
                        
                        print(f"💰 BUY: {shares_to_buy:.2f} shares @ ${price:.2f}")
                
                # Check for sell signal
                elif sma_short < sma_long and self.state.portfolio["shares"] > 0:
                    # Execute sell
                    shares_to_sell = self.state.portfolio["shares"]
                    proceeds = shares_to_sell * price
                    
                    # Calculate P&L
                    entry_value = shares_to_sell * self.state.entry_price
                    pnl = proceeds - entry_value
                    
                    # Update portfolio
                    self.state.portfolio["cash"] += proceeds
                    self.state.portfolio["shares"] = 0
                    
                    # Update stats
                    if pnl > 0:
                        self.state.portfolio["winning_trades"] += 1
                    self.state.portfolio["total_pnl"] += pnl
                    
                    signal = -1
                    trade_amount = shares_to_sell
                    trade_value = proceeds
                    
                    print(f"💸 SELL: {shares_to_sell:.2f} shares @ ${price:.2f} (P&L: ${pnl:+.2f})")
            
            # Record signals
            self.state.signal_history.append(signal)
            self.state.transaction_history.append(trade_amount)
            self.state.transaction_value_history.append(trade_value)
            
            # Update portfolio history
            current_value = self.state.calculate_current_value(price)
            self.state.portfolio["history"].append(current_value)
            
            # Update timestamp
            self.state.last_update = time.time()
    
    def _calculate_sma(self, data_deque: WebSafeDeque, window: int) -> Optional[float]:
        """Calculate Simple Moving Average"""
        with self.state.lock:
            data = data_deque.to_list()
            if len(data) < window:
                return None
            return float(np.mean(data[-window:]))

# ============================================================================
# Web Visualization with Plotly
# ============================================================================

class WebVisualizer:
    """High-performance web visualization using Plotly"""
    
    def __init__(self, engine: WebTradingEngine):
        self.engine = engine
        self.last_update = 0
        self.update_interval = 0.1  # 100ms for smooth updates
        
    def create_dashboard(self):
        """Create Streamlit dashboard"""
        if not WEB_AVAILABLE:
            st.error("Required packages not installed. Run: pip install streamlit plotly")
            return
        
        # Page configuration
        st.set_page_config(
            page_title="Real-Time Trading Dashboard",
            page_icon="📈",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS for better performance
        st.markdown("""
        <style>
        .stPlotlyChart {
            height: 400px !important;
        }
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 10px;
            padding: 20px;
            color: white;
            margin: 10px 0;
        }
        .refresh-warning {
            background-color: #ffdddd;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Title
        st.title("📈 Real-Time Trading Dashboard")
        st.markdown("---")
        
        # Control panel in sidebar
        with st.sidebar:
            st.header("⚙️ Controls")
            
            if st.button("🔄 Restart Engine", use_container_width=True):
                self.engine.stop()
                time.sleep(0.5)
                self.engine.start()
                st.rerun()
            
            if st.button("⏸️ Pause/Resume", use_container_width=True):
                self.engine.state.is_running = not self.engine.state.is_running
                st.rerun()
            
            st.markdown("---")
            st.header("📊 Settings")
            
            # Config sliders
            self.engine.config.short_window = st.slider(
                "Short SMA Window", 5, 50, self.engine.config.short_window
            )
            self.engine.config.long_window = st.slider(
                "Long SMA Window", 20, 100, self.engine.config.long_window
            )
            self.engine.config.buy_allocation = st.slider(
                "Buy Allocation %", 0.1, 1.0, self.engine.config.buy_allocation
            )
            
            st.markdown("---")
            st.header("ℹ️ Info")
            st.info("""
            **Strategy**: Dual Moving Average Crossover
            - Buy when Short SMA crosses above Long SMA
            - Sell when Short SMA crosses below Long SMA
            - Updates every 100ms
            """)
        
        # Main content area
        col1, col2, col3 = st.columns(3)
        
        with col1:
            self._render_metrics_card("💰 Portfolio Value", 
                                    self.engine.state.calculate_current_value(), 
                                    "dollar")
        
        with col2:
            snapshot = self.engine.state.snapshot()
            self._render_metrics_card("📊 Total Trades", 
                                    snapshot["total_trades"], 
                                    "trending-up")
        
        with col3:
            win_rate = (snapshot["total_trades"] > 0 and 
                       self.engine.state.portfolio["winning_trades"] / snapshot["total_trades"] * 100) or 0
            self._render_metrics_card("🎯 Win Rate", 
                                    f"{win_rate:.1f}%", 
                                    "target")
        
        # Charts
        self._render_charts()
        
        # Trading log
        st.markdown("---")
        st.header("📝 Trading Activity")
        self._render_trading_log()
        
        # Auto-refresh
        if self.engine.state.is_running:
            time.sleep(self.update_interval)
            st.rerun()
    
    def _render_metrics_card(self, title: str, value, icon: str):
        """Render a metrics card"""
        st.markdown(f"""
        <div class="metric-card">
            <h3>{title}</h3>
            <h1>{value}</h1>
        </div>
        """, unsafe_allow_html=True)
    
    def _render_charts(self):
        """Render all charts"""
        # Get current snapshot
        snapshot = self.engine.state.snapshot()
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["📈 Price & Signals", "💰 Portfolio", "📊 Performance"])
        
        with tab1:
            self._render_price_chart(snapshot)
        
        with tab2:
            self._render_portfolio_chart(snapshot)
        
        with tab3:
            self._render_performance_chart(snapshot)
    
    def _render_price_chart(self, snapshot: Dict):
        """Render price chart with SMAs and signals"""
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.7, 0.3],
            subplot_titles=("Price & Moving Averages", "Trading Signals")
        )
        
        # Add price trace
        x_axis = list(range(len(snapshot["prices"])))
        
        if snapshot["prices"]:
            fig.add_trace(
                go.Scatter(
                    x=x_axis,
                    y=snapshot["prices"],
                    name="Price",
                    line=dict(color="#00ffff", width=2),
                    mode="lines"
                ),
                row=1, col=1
            )
        
        # Add SMA traces
        if snapshot["sma_short"]:
            fig.add_trace(
                go.Scatter(
                    x=x_axis,
                    y=snapshot["sma_short"],
                    name=f"SMA {self.engine.config.short_window}",
                    line=dict(color="#ffff00", width=1.5, dash="dash"),
                    mode="lines"
                ),
                row=1, col=1
            )
        
        if snapshot["sma_long"]:
            fig.add_trace(
                go.Scatter(
                    x=x_axis,
                    y=snapshot["sma_long"],
                    name=f"SMA {self.engine.config.long_window}",
                    line=dict(color="#ff00ff", width=1.5, dash="dash"),
                    mode="lines"
                ),
                row=1, col=1
            )
        
        # Add buy/sell signals to price chart
        buy_signals = []
        sell_signals = []
        for i, signal in enumerate(snapshot["signals"]):
            if i < len(snapshot["prices"]):
                if signal == 1:
                    buy_signals.append((i, snapshot["prices"][i]))
                elif signal == -1:
                    sell_signals.append((i, snapshot["prices"][i]))
        
        if buy_signals:
            buy_x, buy_y = zip(*buy_signals)
            fig.add_trace(
                go.Scatter(
                    x=buy_x,
                    y=buy_y,
                    name="Buy Signals",
                    mode="markers",
                    marker=dict(color="#00ff00", size=12, symbol="triangle-up"),
                    showlegend=True
                ),
                row=1, col=1
            )
        
        if sell_signals:
            sell_x, sell_y = zip(*sell_signals)
            fig.add_trace(
                go.Scatter(
                    x=sell_x,
                    y=sell_y,
                    name="Sell Signals",
                    mode="markers",
                    marker=dict(color="#ff4444", size=12, symbol="triangle-down"),
                    showlegend=True
                ),
                row=1, col=1
            )
        
        # Add signal strength bars
        if snapshot["signals"]:
            fig.add_trace(
                go.Bar(
                    x=x_axis[:len(snapshot["signals"])],
                    y=snapshot["signals"],
                    name="Signal Strength",
                    marker_color=["green" if s == 1 else "red" if s == -1 else "gray" 
                                 for s in snapshot["signals"]],
                    opacity=0.6
                ),
                row=2, col=1
            )
        
        # Update layout
        fig.update_layout(
            height=600,
            showlegend=True,
            hovermode="x unified",
            template="plotly_dark",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Signal", row=2, col=1)
        fig.update_xaxes(title_text="Time (Ticks)", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})
    
    def _render_portfolio_chart(self, snapshot: Dict):
        """Render portfolio performance chart"""
        fig = go.Figure()
        
        if snapshot["portfolio_value"]:
            # Portfolio value line
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(snapshot["portfolio_value"]))),
                    y=snapshot["portfolio_value"],
                    name="Portfolio Value",
                    line=dict(color="#ffffff", width=3),
                    fill='tozeroy',
                    fillcolor='rgba(255, 255, 255, 0.1)'
                )
            )
            
            # Starting balance line
            fig.add_hline(
                y=self.engine.config.initial_cash,
                line_dash="dash",
                line_color="red",
                annotation_text="Starting Balance",
                annotation_position="bottom right"
            )
            
            # Current value marker
            current_value = snapshot["portfolio_value"][-1] if snapshot["portfolio_value"] else 0
            fig.add_trace(
                go.Scatter(
                    x=[len(snapshot["portfolio_value"]) - 1],
                    y=[current_value],
                    mode="markers+text",
                    marker=dict(size=15, color="yellow"),
                    text=[f"${current_value:,.2f}"],
                    textposition="top center",
                    name="Current Value"
                )
            )
        
        # Update layout
        fig.update_layout(
            title="Portfolio Performance",
            height=500,
            template="plotly_dark",
            hovermode="x unified",
            showlegend=True,
            xaxis_title="Time (Ticks)",
            yaxis_title="Portfolio Value ($)"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Portfolio stats
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Cash Balance", f"${snapshot['current_cash']:,.2f}")
        
        with col2:
            st.metric("Shares Held", f"{snapshot['current_shares']:.2f}")
        
        with col3:
            total_return = ((current_value - self.engine.config.initial_cash) / 
                          self.engine.config.initial_cash * 100)
            st.metric("Total Return", f"{total_return:+.2f}%")
        
        with col4:
            pnl = snapshot["total_pnl"]
            st.metric("Total P&L", f"${pnl:+.2f}")
    
    def _render_performance_chart(self, snapshot: Dict):
        """Render performance metrics chart"""
        # Calculate metrics
        returns = []
        if snapshot["portfolio_value"] and len(snapshot["portfolio_value"]) > 1:
            for i in range(1, len(snapshot["portfolio_value"])):
                if snapshot["portfolio_value"][i-1] != 0:
                    returns.append(
                        (snapshot["portfolio_value"][i] - snapshot["portfolio_value"][i-1]) / 
                        snapshot["portfolio_value"][i-1]
                    )
        
        # Create metrics display
        col1, col2 = st.columns(2)
        
        with col1:
            # Sharpe ratio (simplified)
            sharpe = 0
            if returns and np.std(returns) > 0:
                sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
            
            metrics_data = {
                "Sharpe Ratio": f"{sharpe:.2f}",
                "Total Trades": f"{snapshot['total_trades']}",
                "Win Rate": f"{(snapshot['total_trades'] > 0 and snapshot['total_trades'] / snapshot['total_trades'] * 100) or 0:.1f}%",
                "Max Return": f"{max(returns) * 100 if returns else 0:.2f}%",
            }
            
            for metric, value in metrics_data.items():
                st.metric(metric, value)
        
        with col2:
            # Drawdown chart
            if snapshot["portfolio_value"]:
                portfolio_values = np.array(snapshot["portfolio_value"])
                running_max = np.maximum.accumulate(portfolio_values)
                drawdown = (portfolio_values - running_max) / running_max * 100
                
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(drawdown))),
                        y=drawdown,
                        name="Drawdown",
                        fill='tozeroy',
                        fillcolor='rgba(255, 0, 0, 0.3)',
                        line=dict(color='red')
                    )
                )
                
                fig.update_layout(
                    title="Portfolio Drawdown",
                    height=300,
                    template="plotly_dark",
                    xaxis_title="Time (Ticks)",
                    yaxis_title="Drawdown (%)",
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    def _render_trading_log(self):
        """Render trading activity log"""
        snapshot = self.engine.state.snapshot()
        
        # Create a DataFrame of recent trades
        trades = []
        for i, signal in enumerate(snapshot["signals"]):
            if signal != 0 and i < len(snapshot["prices"]):
                trade_type = "BUY" if signal == 1 else "SELL"
                price = snapshot["prices"][i]
                value = snapshot["transaction_value_history"].to_list()[i] if i < len(snapshot["transaction_value_history"].to_list()) else 0
                
                trades.append({
                    "Time": f"T-{len(snapshot['signals']) - i}",
                    "Type": trade_type,
                    "Price": f"${price:.2f}",
                    "Value": f"${value:.2f}",
                    "Signal": "🟢" if signal == 1 else "🔴"
                })
        
        # Show last 10 trades
        if trades:
            df_trades = pd.DataFrame(trades[-10:][::-1])  # Show most recent first
            st.dataframe(df_trades, use_container_width=True, hide_index=True)
        else:
            st.info("No trades yet. Waiting for signals...")

# ============================================================================
# Streamlit App Entry Point
# ============================================================================

def main():
    """Main Streamlit application"""
    if not WEB_AVAILABLE:
        print("Please install required packages: pip install streamlit plotly")
        return
    
    # Initialize trading engine
    config = TradingConfig(
        symbol="SPY",
        initial_cash=10000.00,
        stream_mode=StreamMode.MOCK
    )
    
    # Create engine and start it
    if 'engine' not in st.session_state:
        st.session_state.engine = WebTradingEngine(config)
        st.session_state.engine.start()
    
    # Create visualizer
    visualizer = WebVisualizer(st.session_state.engine)
    
    # Run dashboard
    visualizer.create_dashboard()