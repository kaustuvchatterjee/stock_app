"""
Chart Module for Stock Market Application
Handles chart creation and visualization using Plotly
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import json


class ChartModule:
    def __init__(self):
        """Initialize the chart module"""
        self.default_layout = {
            "height": 600,
            "showlegend": False,
            "xaxis": {"rangeslider": {"visible": False}},
            "margin": {"l": 50, "r": 50, "t": 50, "b": 50},
            "template": "plotly_white"
        }
    
    def create_historical_figure(self, data: pd.DataFrame, pchange: float, 
                                portfolio_holdings: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Create historical chart with technical indicators
        
        Args:
            data: DataFrame with historical data and indicators
            pchange: Percentage change from previous close
            portfolio_holdings: Optional dict with quantity and avg_price for portfolio visualization
            
        Returns:
            Plotly figure as JSON string for embedding in QWebEngineView
        """
        if data.empty:
            # Return empty figure if no data
            fig = go.Figure()
            fig.update_layout(title="No data available")
            return json.loads(fig.to_json())
        
        # Ensure data is properly formatted
        data_copy = data.copy()
        
        # Convert all numeric columns to float to ensure proper plotting
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'upper_bound', 'lower_bound', 
                       'sma', 'rsi', 'MACD_Histo']
        for col in numeric_cols:
            if col in data_copy.columns:
                data_copy[col] = data_copy[col].astype(float)
        
        # Check for NaN values in MACD_Histo and replace with 0
        if 'MACD_Histo' in data_copy.columns:
            data_copy['MACD_Histo'] = data_copy['MACD_Histo'].fillna(0)
        
        # Calculate MACD range for proper scaling
        if 'MACD_Histo' in data_copy.columns:
            macd_max = data_copy['MACD_Histo'].abs().max()
            # Add a small buffer to prevent bars from touching axis limits
            macd_range = [-macd_max * 1.1, macd_max * 1.1]
        else:
            macd_range = [-1, 1]  # Default range if no MACD data

        # Create subplots - exactly as in algotrade.py
        fig = make_subplots(rows=3, cols=1, row_heights=[0.6, 0.2, 0.2], 
                           vertical_spacing=0, shared_xaxes=True)
        
        # Define traces exactly as in algotrade.py
        # Upper Bollinger Band
        ub_plot = go.Scatter(
            x=data_copy['Date'].tolist(),
            y=data_copy['upper_bound'].tolist(),
            mode='lines',
            line=dict(color='lightblue', width=0),
            name='Upper Bound'
        )
        
        # Lower Bollinger Band (with fill between)
        lb_plot = go.Scatter(
            x=data_copy['Date'].tolist(),
            y=data_copy['lower_bound'].tolist(),
            mode='lines',
            line=dict(color='lightblue', width=0),
            fill='tonexty',
            fillcolor='rgba(31, 119, 180, 0.1)',
            name='Lower Bound'
        )
        
        # SMA line
        sma_plot = go.Scatter(
            x=data_copy['Date'].tolist(),
            y=data_copy['sma'].tolist(),
            mode='lines',
            line=dict(color='#1f77b4', width=1),
            name='SMA'
        )
        
        # Close price line
        close_plot = go.Scatter(
            x=data_copy['Date'].tolist(),
            y=data_copy['Close'].tolist(),
            mode='lines',
            line=dict(color='lightgray', width=1),
            name='Close'
        )
        
        # Candlestick chart
        candle_stick = go.Candlestick(
            x=data_copy['Date'].tolist(),
            open=data_copy['Open'].tolist(),
            close=data_copy['Close'].tolist(),
            high=data_copy['High'].tolist(),
            low=data_copy['Low'].tolist(),
            name='Candlestick'
        )
        
        # RSI plot
        rsi_plot = go.Scatter(
            x=data_copy['Date'].tolist(),
            y=data_copy['rsi'].tolist(),
            mode='lines',
            line=dict(color='red', width=2),
            name='RSI'
        )
        
        # MACD histogram - ensure data is properly formatted
        macd_values = data_copy['MACD_Histo'].tolist()
        date_values = data_copy['Date'].tolist()
        color_values = data_copy['Color'].tolist()
        
        # Create MACD plot exactly as in algotrade.py
        macd_plot = go.Bar(
            x=date_values,
            y=macd_values,
            marker_color=color_values,
            name='MACD'
        )
        
        # Group traces by subplot exactly as in algotrade.py
        plot1 = [ub_plot, lb_plot, sma_plot, close_plot, candle_stick]
        plot2 = [rsi_plot]
        plot3 = [macd_plot]
        
        # Add traces to subplots in the exact same way as algotrade.py
        fig.add_traces(plot1, rows=1, cols=1)
        fig.add_traces(plot2, rows=2, cols=1)
        fig.add_traces(plot3, rows=3, cols=1)
        
        # Add title annotation
        fig.add_annotation(
            x=0,
            y=1,
            text=data_copy['ticker_name'][0] if 'ticker_name' in data_copy.columns else "Stock Chart",
            font=dict(size=20),
            showarrow=False,
            xanchor='left',
            yanchor='bottom',
            xref='x domain',
            yref='y domain',
            row=1,
            col=1,
        )
        
        # Add price reference lines
        if pchange >= 0:
            # Previous close line
            prev_close = float(data_copy.iloc[-2]['Close'])
            fig.add_hline(y=prev_close, line_color='black', line_width=0.3)
            fig.add_annotation(
                x=data_copy.iloc[-1]['Date'],
                y=prev_close,
                text=f"{prev_close:.2f}",
                showarrow=False,
                xanchor="left",
                yanchor='top'
            )
            
            # Current close line
            curr_close = float(data_copy.iloc[-1]['Close'])
            fig.add_hline(y=curr_close, line_color='green', line_width=0.3)
            fig.add_annotation(
                x=data_copy.iloc[-1]['Date'],
                y=curr_close,
                text=f"{curr_close:.2f}",
                font=dict(color="green"),
                showarrow=False,
                xanchor="left",
                yanchor='bottom'
            )
        else:
            # Previous close line
            prev_close = float(data_copy.iloc[-2]['Close'])
            fig.add_hline(y=prev_close, line_color='black', line_width=0.3)
            fig.add_annotation(
                x=data_copy.iloc[-1]['Date'],
                y=prev_close,
                text=f"{prev_close:.2f}",
                showarrow=False,
                xanchor="left",
                yanchor='bottom'
            )
            
            # Current close line
            curr_close = float(data_copy.iloc[-1]['Close'])
            fig.add_hline(y=curr_close, line_color='red', line_width=0.3)
            fig.add_annotation(
                x=data_copy.iloc[-1]['Date'],
                y=curr_close,
                text=f"{curr_close:.2f}",
                font=dict(color="red"),
                showarrow=False,
                xanchor="left",
                yanchor='top'
            )
        
        # Add portfolio holdings visualization if provided
        if portfolio_holdings and 'avg_price' in portfolio_holdings and 'quantity' in portfolio_holdings:
            avg_price = float(portfolio_holdings['avg_price'])
            quantity = float(portfolio_holdings['quantity'])
            
            # Add horizontal line at average purchase price
            fig.add_hline(
                y=avg_price, 
                line_color='blue', 
                line_width=1.5, 
                line_dash='dash',
                row=1, 
                col=1
            )
            
            # Add annotation for holdings
            fig.add_annotation(
                x=data_copy.iloc[-1]['Date'],
                y=avg_price,
                text=f"Holdings: {quantity:.2f} @ {avg_price:.2f}",
                font=dict(color="blue", size=12),
                showarrow=True,
                arrowhead=2,
                arrowcolor="blue",
                arrowsize=1,
                arrowwidth=1,
                ax=0,
                ay=-40,
                row=1,
                col=1
            )
            
            # Color the background to indicate profit/loss zone
            current_price = float(data_copy.iloc[-1]['Close'])
            if current_price > avg_price:
                # Profit zone (light green)
                fig.add_shape(
                    type="rect",
                    x0=data_copy.iloc[0]['Date'],
                    x1=data_copy.iloc[-1]['Date'],
                    y0=avg_price,
                    y1=current_price,
                    fillcolor="rgba(0,255,0,0.1)",
                    line=dict(width=0),
                    layer="below",
                    row=1,
                    col=1
                )
            elif current_price < avg_price:
                # Loss zone (light red)
                fig.add_shape(
                    type="rect",
                    x0=data_copy.iloc[0]['Date'],
                    x1=data_copy.iloc[-1]['Date'],
                    y0=current_price,
                    y1=avg_price,
                    fillcolor="rgba(255,0,0,0.1)",
                    line=dict(width=0),
                    layer="below",
                    row=1,
                    col=1
                )
        
        # Add vertical lines for signal crossings - for all subplots
        for i in range(len(data_copy)):
            if (data_copy.iloc[i]['z_cross'] == 1) or (data_copy.iloc[i]['z_cross'] == -1):
                # Add to all three subplots
                fig.add_vline(data_copy.iloc[i]['Date'], line_color='lightgray', line_width=0.3)
        
        # Add vertical lines for trade signals - for all subplots
        for i in range(len(data_copy)):
            if data_copy.iloc[i]['trade_signal'] == 1:
                # Add to all three subplots
                fig.add_vline(data_copy.iloc[i]['Date'], line_color='red', line_width=0.8)
            if data_copy.iloc[i]['trade_signal'] == -1:
                # Add to all three subplots
                fig.add_vline(data_copy.iloc[i]['Date'], line_color='green', line_width=0.8)
        
        # Add RSI reference lines
        fig.add_hline(y=70, line_color='red', line_dash='dash', row=2, col=1)
        fig.add_hline(y=30, line_color='teal', line_dash='dash', row=2, col=1)
        
        # Add RSI title
        fig.add_annotation(
            x=0,
            y=1,
            text="RSI",
            font=dict(size=14),
            showarrow=False,
            xanchor='left',
            xref='x domain',
            yref='y domain',
            row=2,
            col=1,
        )
        
        # Add MACD title
        fig.add_annotation(
            x=0,
            y=1,
            text="MACD",
            font=dict(size=14),
            showarrow=False,
            xanchor='left',
            xref='x domain',
            yref='y domain',
            row=3,
            col=1,
        )
        
        # Add zero line for MACD
        fig.add_hline(y=0, line_color='black', line_width=0.5, row=3, col=1)
        
        # Update layout - with proper scaling for MACD
        layout = {
            "height": 600,
            "showlegend": False,
            "xaxis": {"rangeslider": {"visible": False}},
            "yaxis2": {"range": [0, 100]},
            "yaxis3": {"range": macd_range, "showticklabels": False},  # Set explicit range for MACD
        }
        fig.update_layout(layout)
        
        # Convert to JSON for embedding in QWebEngineView
        return json.loads(fig.to_json())
    
    def create_current_figure(self, live_data: pd.DataFrame, 
                             portfolio_holdings: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Create current day chart with volume
        
        Args:
            live_data: DataFrame with current day data
            portfolio_holdings: Optional dict with quantity and avg_price for portfolio visualization
            
        Returns:
            Plotly figure as JSON string for embedding in QWebEngineView
        """
        if live_data.empty:
            # Return empty figure if no data
            fig = go.Figure()
            fig.update_layout(title="No current day data available")
            return json.loads(fig.to_json())
        
        # Ensure data is properly formatted
        live_data_copy = live_data.copy()
        
        # Convert all numeric columns to float to ensure proper plotting
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_cols:
            if col in live_data_copy.columns:
                live_data_copy[col] = live_data_copy[col].astype(float)
        
        # Create figure with secondary y-axis for volume - exactly as in algotrade.py
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Candlestick chart for price
        today_plot = go.Candlestick(
            x=live_data_copy['Datetime'].tolist(),
            open=live_data_copy['Open'].tolist(),
            close=live_data_copy['Close'].tolist(),
            high=live_data_copy['High'].tolist(),
            low=live_data_copy['Low'].tolist()
        )
        
        # Bar chart for volume
        vol_plot = go.Bar(
            x=live_data_copy['Datetime'].tolist(),
            y=live_data_copy['Volume'].tolist(),
            name='Volume',
            marker_color='rgba(0,0,255,0.3)'
        )
        
        fig.add_trace(today_plot, secondary_y=False)
        fig.add_trace(vol_plot, secondary_y=True)
        
        # Add portfolio holdings visualization if provided
        if portfolio_holdings and 'avg_price' in portfolio_holdings and 'quantity' in portfolio_holdings:
            avg_price = float(portfolio_holdings['avg_price'])
            quantity = float(portfolio_holdings['quantity'])
            
            # Add horizontal line at average purchase price
            fig.add_hline(
                y=avg_price, 
                line_color='blue', 
                line_width=1.5, 
                line_dash='dash'
            )
            
            # Add annotation for holdings
            fig.add_annotation(
                x=live_data_copy.iloc[-1]['Datetime'],
                y=avg_price,
                text=f"Holdings: {quantity:.2f} @ {avg_price:.2f}",
                font=dict(color="blue", size=12),
                showarrow=True,
                arrowhead=2,
                arrowcolor="blue",
                arrowsize=1,
                arrowwidth=1,
                ax=0,
                ay=-40
            )
        
        # Update layout - exactly as in algotrade.py
        layout = {
            "height": 600,
            "showlegend": False,
            "xaxis": {"rangeslider": {"visible": False}},
            "yaxis2": {
                "showticklabels": False,
                "showgrid": False,
                "range": [0, live_data_copy['Volume'].max() * 10 if not live_data_copy['Volume'].empty else 1000]
            }
        }
        fig.update_layout(layout)
        
        # Convert to JSON for embedding in QWebEngineView
        return json.loads(fig.to_json())
    
    def create_portfolio_figure(self, holdings_data: pd.DataFrame, 
                               performance_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Create portfolio performance chart
        
        Args:
            holdings_data: DataFrame with current holdings
            performance_data: Optional DataFrame with historical portfolio value
            
        Returns:
            Plotly figure as JSON string for embedding in QWebEngineView
        """
        if holdings_data.empty:
            # Return empty figure if no holdings
            fig = go.Figure()
            fig.update_layout(title="No portfolio holdings")
            return json.loads(fig.to_json())
        
        # Create figure
        fig = go.Figure()
        
        # Add pie chart for holdings allocation
        labels = holdings_data['ticker'].tolist()
        values = (holdings_data['quantity'] * holdings_data['current_price']).tolist()
        
        fig.add_trace(go.Pie(
            labels=labels,
            values=values,
            textinfo='label+percent',
            hole=0.4,
            marker=dict(
                line=dict(color='#FFFFFF', width=2)
            )
        ))
        
        # Add performance chart if data is provided
        if performance_data is not None and not performance_data.empty:
            # This would be implemented if we had historical portfolio value data
            pass
        
        # Update layout
        layout = self.default_layout.copy()
        layout.update({
            "title": "Portfolio Allocation",
            "showlegend": True,
            "legend": dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        })
        fig.update_layout(layout)
        
        # Convert to JSON for embedding in QWebEngineView
        return json.loads(fig.to_json())
