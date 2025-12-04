import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import time
import random
import threading
import datetime
import math
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Optional
from collections import deque
from enum import Enum
from contextlib import contextmanager
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================================================
# Configuration & Constants
# ============================================================================

class StreamMode(Enum):
    MOCK = "mock"
    ALPACA = "alpaca"

@dataclass
class TradingConfig:
    """Central configuration for trading system"""
    api_key: Optional[str] = field(default_factory=lambda: os.getenv("ALPACA_API_KEY"))
    secret_key: Optional[str] = field(default_factory=lambda: os.getenv("ALPACA_SECRET_KEY"))
    
    symbol: str = "SPY"
    short_window: int = 10
    long_window: int = 30
    max_data_points: int = 200
    buy_allocation: float = 0.5
    initial_cash: float = 1000.00
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.10
    risk_per_trade: float = 0.02
    stream_mode: StreamMode = StreamMode.MOCK

# ============================================================================
# Global State
# ============================================================================

class TradingState:
    def __init__(self, config: TradingConfig):
        self.config = config
        self.lock = threading.RLock()
        self.price_data = deque(maxlen=config.max_data_points)
        self.sma_short_data = deque(maxlen=config.max_data_points)
        self.sma_long_data = deque(maxlen=config.max_data_points)
        self.timestamps = deque(maxlen=config.max_data_points)
        self.signal_history = deque(maxlen=config.max_data_points)
        self.transaction_history = deque(maxlen=config.max_data_points)
        self.transaction_value_history = deque(maxlen=config.max_data_points)
        
        self.portfolio = {
            "cash": config.initial_cash,
            "shares": 0.0,
            "history": deque(maxlen=config.max_data_points),
            "total_trades": 0,
            "winning_trades": 0,
            "total_pnl": 0.0
        }
        self.entry_price = 0.0
        self.high_water_mark = config.initial_cash
        self.max_drawdown = 0.0

    @contextmanager
    def atomic(self):
        with self.lock:
            yield

    def calculate_current_value(self, current_price: float) -> float:
        return self.portfolio["cash"] + (self.portfolio["shares"] * current_price)

# ============================================================================
# Logic & Engine
# ============================================================================

class MovingAverageStrategy:
    def __init__(self, config: TradingConfig, state: TradingState):
        self.config = config
        self.state = state
        
    def calculate_sma(self, window: int) -> Optional[float]:
        with self.state.atomic():
            if len(self.state.price_data) < window:
                return None
            prices = np.array(list(self.state.price_data)[-window:])
            return float(np.mean(prices))
    
    def calculate_position_size(self, price: float) -> float:
        with self.state.atomic():
            risk_amount = self.state.portfolio["cash"] * self.config.risk_per_trade
            volatility = 0.01 
            if len(self.state.price_data) >= 20:
                recent = list(self.state.price_data)[-20:]
                rets = np.diff(recent) / recent[:-1]
                if len(rets) > 0: volatility = float(np.std(rets))
            
            position = risk_amount / (price * volatility) if volatility > 0 else risk_amount / price
            max_pos = (self.state.portfolio["cash"] * self.config.buy_allocation) / price
            return min(position, max_pos)

    def check_stop_loss(self, current_price: float) -> bool:
        with self.state.atomic():
            if self.state.portfolio["shares"] == 0 or self.state.entry_price == 0: return False
            loss_pct = (self.state.entry_price - current_price) / self.state.entry_price
            return loss_pct >= self.config.stop_loss_pct
    
    def check_take_profit(self, current_price: float) -> bool:
        with self.state.atomic():
            if self.state.portfolio["shares"] == 0 or self.state.entry_price == 0: return False
            profit_pct = (current_price - self.state.entry_price) / self.state.entry_price
            return profit_pct >= self.config.take_profit_pct

class TradingEngine:
    def __init__(self, config: TradingConfig):
        self.config = config
        self.state = TradingState(config)
        self.strategy = MovingAverageStrategy(config, self.state)
        
    def process_market_event(self, data: dict):
        try:
            if not data or 'p' not in data: return
            timestamp = data.get('t', datetime.datetime.now().isoformat())
            price = float(data['p'])
            
            with self.state.atomic():
                self.state.price_data.append(price)
                self.state.timestamps.append(timestamp)
                
                sma_short = self.strategy.calculate_sma(self.config.short_window)
                sma_long = self.strategy.calculate_sma(self.config.long_window)
                
                self.state.sma_short_data.append(sma_short)
                self.state.sma_long_data.append(sma_long)
                
                signal = 0
                trade_val = 0.0
                
                if sma_short and sma_long:
                    should_exit = False
                    if self.strategy.check_stop_loss(price) or self.strategy.check_take_profit(price):
                        should_exit = True
                    
                    if should_exit and self.state.portfolio["shares"] > 0:
                        self._execute_sell(price)
                        signal = -1
                    elif sma_short > sma_long and self.state.portfolio["shares"] == 0:
                        self._execute_buy(price)
                        signal = 1
                    elif sma_short < sma_long and self.state.portfolio["shares"] > 0:
                        self._execute_sell(price)
                        signal = -1
                
                self.state.signal_history.append(signal)
                current_val = self.state.calculate_current_value(price)
                self.state.portfolio["history"].append(current_val)
                
                # Update Max DD
                if current_val > self.state.high_water_mark:
                    self.state.high_water_mark = current_val
                dd = (self.state.high_water_mark - current_val) / self.state.high_water_mark
                if dd > self.state.max_drawdown: self.state.max_drawdown = dd

        except Exception as e:
            print(f"Error: {e}")

    def _execute_buy(self, price: float):
        shares = self.strategy.calculate_position_size(price)
        cost = shares * price
        if cost > self.state.portfolio["cash"]:
            shares = self.state.portfolio["cash"] / price
            cost = self.state.portfolio["cash"]
        
        if shares > 0:
            self.state.portfolio["shares"] += shares
            self.state.portfolio["cash"] -= cost
            self.state.entry_price = price
            self.state.portfolio["total_trades"] += 1

    def _execute_sell(self, price: float):
        if self.state.portfolio["shares"] == 0: return
        proceeds = self.state.portfolio["shares"] * price
        entry_val = self.state.portfolio["shares"] * self.state.entry_price
        pnl = proceeds - entry_val
        
        self.state.portfolio["cash"] += proceeds
        self.state.portfolio["shares"] = 0
        if pnl > 0: self.state.portfolio["winning_trades"] += 1
        self.state.portfolio["total_pnl"] += pnl
        self.state.entry_price = 0.0

# ============================================================================
# Streams
# ============================================================================

class MockDataStream:
    def __init__(self, symbol, callback):
        self.symbol = symbol
        self.callback = callback
        self.running = False
        self.price = 100.0
        
    def start(self):
        self.running = True
        threading.Thread(target=self._run, daemon=True).start()
        
    def stop(self):
        self.running = False
        
    def _run(self):
        tick = 0
        while self.running:
            tick += 1
            trend = math.sin(tick * 0.1) * 0.5
            noise = random.uniform(-0.2, 0.2)
            self.price += trend + noise
            self.price = max(10, self.price)
            
            data = {
                "T": "t", "S": self.symbol, "p": self.price,
                "s": 100, "t": datetime.datetime.now().isoformat()
            }
            self.callback(data)
            time.sleep(0.1)

# ============================================================================
# Streamlit Interface
# ============================================================================

def init_session_state():
    if 'engine' not in st.session_state:
        config = TradingConfig(stream_mode=StreamMode.MOCK)
        st.session_state.engine = TradingEngine(config)
        st.session_state.stream = MockDataStream("SPY", st.session_state.engine.process_market_event)
        st.session_state.stream_active = False

def render_dashboard():
    st.set_page_config(layout="wide", page_title="Trading Bot")
    init_session_state()
    
    st.title("⚡ Real-Time Trading Bot")
    
    # Control Panel
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("Start Bot"):
            if not st.session_state.stream_active:
                st.session_state.stream.start()
                st.session_state.stream_active = True
    with col2:
        if st.button("Stop Bot"):
            if st.session_state.stream_active:
                st.session_state.stream.stop()
                st.session_state.stream_active = False
    
    # Live Metrics
    placeholder_metrics = st.empty()
    placeholder_chart = st.empty()
    
    while st.session_state.stream_active:
        engine = st.session_state.engine
        state = engine.state
        
        with state.atomic():
            if len(state.price_data) > 0:
                current_price = state.price_data[-1]
                equity = state.calculate_current_value(current_price)
                cash = state.portfolio["cash"]
                shares = state.portfolio["shares"]
                pnl = equity - state.config.initial_cash
                
                # Render Metrics
                with placeholder_metrics.container():
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Equity", f"${equity:,.2f}", f"{pnl:+.2f}")
                    m2.metric("Cash", f"${cash:,.2f}")
                    m3.metric("Shares", f"{shares:.4f}")
                    m4.metric("Trades", f"{state.portfolio['total_trades']}")

                # Render Chart using Plotly
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                   vertical_spacing=0.05, row_heights=[0.7, 0.3])
                
                # Price & SMAs
                x_axis = list(range(len(state.price_data)))
                fig.add_trace(go.Scatter(x=x_axis, y=list(state.price_data), 
                                       name="Price", line=dict(color='cyan')), row=1, col=1)
                
                sma_s = list(state.sma_short_data)
                sma_l = list(state.sma_long_data)
                fig.add_trace(go.Scatter(x=x_axis, y=[v if v else None for v in sma_s], 
                                       name="SMA Fast", line=dict(color='yellow')), row=1, col=1)
                fig.add_trace(go.Scatter(x=x_axis, y=[v if v else None for v in sma_l], 
                                       name="SMA Slow", line=dict(color='magenta')), row=1, col=1)
                
                # Buy/Sell Markers
                signals = list(state.signal_history)
                buys_x = [i for i, s in enumerate(signals) if s == 1]
                buys_y = [state.price_data[i] for i in buys_x]
                sells_x = [i for i, s in enumerate(signals) if s == -1]
                sells_y = [state.price_data[i] for i in sells_x]
                
                fig.add_trace(go.Scatter(x=buys_x, y=buys_y, mode='markers', 
                                       marker=dict(symbol='triangle-up', size=12, color='green'),
                                       name='Buy'), row=1, col=1)
                fig.add_trace(go.Scatter(x=sells_x, y=sells_y, mode='markers', 
                                       marker=dict(symbol='triangle-down', size=12, color='red'),
                                       name='Sell'), row=1, col=1)

                # Equity Curve
                history = list(state.portfolio["history"])
                # Pad history to match x_axis length if needed
                if len(history) < len(x_axis):
                    history = [None]*(len(x_axis)-len(history)) + history
                
                fig.add_trace(go.Scatter(x=x_axis, y=history, 
                                       name="Equity", line=dict(color='white')), row=2, col=1)

                fig.update_layout(height=600, template="plotly_dark", margin=dict(l=0, r=0, t=30, b=0))
                placeholder_chart.plotly_chart(fig, use_container_width=True)
        
        time.sleep(0.1)

if __name__ == "__main__":
    render_dashboard()
