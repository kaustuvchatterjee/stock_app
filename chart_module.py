"""
Chart Module for Stock Market Application
Handles chart creation and visualization using Plotly
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import json
from scipy.signal import argrelextrema


class ChartModule:
    def __init__(self):
        """Initialize the chart module"""
        self.default_layout = {
            "height": 400,
            "showlegend": False,
            "xaxis": {"rangeslider": {"visible": False}},
            "margin": {"l": 50, "r": 50, "t": 50, "b": 50},
            "template": "plotly_white"
        }

    def _calculate_rsi(self, data: pd.DataFrame, window: int = 14) -> pd.Series:
        """
        Calculates the Relative Strength Index (RSI).
        Private helper method.

        Args:
            data (pd.DataFrame): DataFrame with a 'Close' column.
            window (int): The RSI calculation window (default is 14).

        Returns:
            pd.Series: A Series containing RSI values.
        """
        if 'Close' not in data.columns or data['Close'].isnull().all():
            # print("DEBUG: _calculate_rsi: 'Close' column not found or is all NaN.") # Removed debug
            return pd.Series(dtype='float64')

        delta = data['Close'].diff()
        gain = delta.copy()
        loss = delta.copy()

        gain[gain < 0] = 0
        loss[loss > 0] = 0

        avg_gain = gain.rolling(window=window, min_periods=1).mean()
        avg_loss = abs(loss).rolling(window=window, min_periods=1).mean()

        with np.errstate(divide='ignore', invalid='ignore'):
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        return rsi

    def _detect_rsi_divergence(self, data: pd.DataFrame, rsi: pd.Series, window: int = 10) -> Tuple[List[Tuple[pd.Timestamp, pd.Timestamp]], List[Tuple[pd.Timestamp, pd.Timestamp]]]:
        """
        Detects bullish and bearish RSI divergences.
        Private helper method.

        Args:
            data (pd.DataFrame): DataFrame with 'Close' prices (expected to have DatetimeIndex).
            rsi (pd.Series): Calculated RSI values (expected to have DatetimeIndex).
            window (int): Window for finding local extrema.

        Returns:
            Tuple[List[Tuple[pd.Timestamp, pd.Timestamp]], List[Tuple[pd.Timestamp, pd.Timestamp]]]:
            Lists of detected bullish and bearish divergence points (start_date, end_date).
        """
        # Ensure data and rsi are aligned and free of NaNs for extrema detection
        # combined_data will inherit the DatetimeIndex from data/rsi
        combined_data = pd.DataFrame({'Close': data['Close'], 'RSI': rsi}).dropna()

        if combined_data.empty or len(combined_data) < window * 2:
            # print(f"DEBUG: _detect_rsi_divergence: Insufficient data after dropping NaNs or for window {window}.") # Removed debug
            return [], []

        # Find local minima/maxima for price and RSI
        price_lows_relative_idx = argrelextrema(combined_data['Close'].values, np.less_equal, order=window)[0]
        price_highs_relative_idx = argrelextrema(combined_data['Close'].values, np.greater_equal, order=window)[0]

        # Convert relative indices to actual DataFrame indices (dates)
        price_lows_dates = combined_data.index[price_lows_relative_idx]
        price_highs_dates = combined_data.index[price_highs_relative_idx]

        bullish_divergences = []
        bearish_divergences = []

        # --- Bullish Divergence: Price Lower Low, RSI Higher Low ---
        for i in range(len(price_lows_dates) - 1):
            p1_date = price_lows_dates[i]
            p2_date = price_lows_dates[i+1]

            price_at_p1 = combined_data.loc[p1_date]['Close']
            price_at_p2 = combined_data.loc[p2_date]['Close']

            if price_at_p2 < price_at_p1: # Price makes a lower low
                rsi_at_p1 = combined_data.loc[p1_date]['RSI']
                rsi_at_p2 = combined_data.loc[p2_date]['RSI']

                if rsi_at_p2 > rsi_at_p1: # RSI makes a higher low at these corresponding points
                    bullish_divergences.append((p1_date, p2_date))

        # --- Bearish Divergence: Price Higher High, RSI Lower High ---
        for i in range(len(price_highs_dates) - 1):
            p1_date = price_highs_dates[i]
            p2_date = price_highs_dates[i+1]

            price_at_p1 = combined_data.loc[p1_date]['Close']
            price_at_p2 = combined_data.loc[p2_date]['Close']

            if price_at_p2 > price_at_p1: # Price makes a higher high
                rsi_at_p1 = combined_data.loc[p1_date]['RSI']
                rsi_at_p2 = combined_data.loc[p2_date]['RSI']

                if rsi_at_p2 < rsi_at_p1: # RSI makes a lower high at these corresponding points
                    bearish_divergences.append((p1_date, p2_date))

        return bullish_divergences, bearish_divergences

    def create_historical_figure(self, data: pd.DataFrame, pchange: float,
                                portfolio_holdings: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Create historical chart with technical indicators including RSI divergence.

        Args:
            data: DataFrame with historical data and indicators.
            pchange: Percentage change from previous close.
            portfolio_holdings: Optional dict with quantity and avg_price for portfolio visualization.

        Returns:
            Plotly figure as JSON string for embedding in QWebEngineView.
        """
        if data.empty:
            # print("DEBUG: create_historical_figure: Input data is empty.") # Removed debug
            fig = go.Figure()
            fig.update_layout(title="No data available")
            return json.loads(fig.to_json())

        data_copy = data.copy()

        # Ensure 'Date' column exists. If index is DatetimeIndex, create 'Date' column from it.
        if 'Date' not in data_copy.columns and isinstance(data_copy.index, pd.DatetimeIndex):
            data_copy['Date'] = data_copy.index.strftime('%Y-%m-%d')
        elif 'Date' not in data_copy.columns:
            print("WARNING: create_historical_figure: 'Date' column missing and index is not DatetimeIndex. Using index as string.")
            data_copy['Date'] = data_copy.index.astype(str)


        # Convert all relevant numeric columns to float, coercing errors
        numeric_cols_to_convert = ['Open', 'High', 'Low', 'Close', 'Volume',
                                   'short', 'long', 'MACD', 'Signal', 'MACD_Histo',
                                   'sma', 'stddev', 'upper_bound', 'lower_bound', 'rsi']
        for col in numeric_cols_to_convert:
            if col in data_copy.columns:
                data_copy[col] = pd.to_numeric(data_copy[col], errors='coerce')
            else:
                # Add missing columns with NaN to avoid KeyError later, then fill
                data_copy[col] = np.nan

        # Fill NaNs for all columns that will be plotted as lines/bars
        # For indicators, forward fill first, then fill any remaining leading NaNs with 0 or a neutral value
        for col in ['Open', 'High', 'Low', 'Close', 'Volume',
                     'short', 'long', 'MACD', 'Signal', 'MACD_Histo',
                     'sma', 'stddev', 'upper_bound', 'lower_bound', 'rsi']:
            if col in data_copy.columns:
                data_copy[col] = data_copy[col].fillna(0) # Fill leading NaNs with 0

        # Special handling for RSI: fill with 50 (neutral) if still NaN
        if 'rsi' in data_copy.columns:
            data_copy['rsi'] = data_copy['rsi'].fillna(50)

        # MACD Histogram Color (ensure it's generated after MACD_Histo is filled)
        if 'MACD_Histo' in data_copy.columns:
            data_copy['Color'] = np.where(data_copy['MACD_Histo'] < 0, '#EF5350', '#26A69A')
        else:
            data_copy['Color'] = '#26A69A' # Default color if MACD_Histo is not available/valid


        # Calculate MACD range for proper scaling
        macd_range = [-1, 1]
        if 'MACD_Histo' in data_copy.columns and not data_copy['MACD_Histo'].empty:
            macd_max = data_copy['MACD_Histo'].abs().max()
            if not np.isnan(macd_max) and macd_max > 0:
                macd_range = [-macd_max * 1.1, macd_max * 1.1]
            else:
                macd_range = [-1, 1] # Fallback if max is 0 or NaN

        # Create subplots
        fig = make_subplots(rows=3, cols=1, row_heights=[0.6, 0.2, 0.2],
                           vertical_spacing=0, shared_xaxes=True)

        # Row 1: Main Price Chart (Candlestick, Bollinger Bands, SMA)
        if all(col in data_copy.columns for col in ['Open', 'High', 'Low', 'Close']) and \
           not data_copy[['Open', 'High', 'Low', 'Close']].isnull().all().any():
            fig.add_trace(go.Candlestick(
                x=data_copy['Date'].tolist(),
                open=data_copy['Open'].astype(float).tolist(),
                close=data_copy['Close'].astype(float).tolist(),
                high=data_copy['High'].astype(float).tolist(),
                low=data_copy['Low'].astype(float).tolist(),
                name='Candlestick', showlegend=False
            ), row=1, col=1)

            # Close price line
            fig.add_trace(go.Scatter(
                x=data_copy['Date'].tolist(),
                y=data_copy['Close'].tolist(),
                mode='lines',
                line=dict(color='lightgray', width=1),
                name='Close', showlegend=False
            ), row=1, col=1)
        else:
            print("WARNING: Not all OHLC data is valid for candlestick plot. Plotting Close price as line.")
            if 'Close' in data_copy.columns and not data_copy['Close'].isnull().all():
                fig.add_trace(go.Scatter(
                    x=data_copy['Date'].tolist(), y=data_copy['Close'].astype(float).tolist(), mode='lines',
                    line=dict(color='blue'), name='Close Price', showlegend=False
                ), row=1, col=1)
            else:
                print("ERROR: 'Close' column also invalid for line plot.")


        if 'upper_bound' in data_copy.columns and 'lower_bound' in data_copy.columns and \
           not data_copy[['upper_bound', 'lower_bound']].isnull().all().any():
            fig.add_trace(go.Scatter(
                x=data_copy['Date'].tolist(), y=data_copy['upper_bound'].astype(float).tolist(), mode='lines',
                line=dict(color='lightblue', width=0), name='Upper Bound', showlegend=False
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=data_copy['Date'].tolist(), y=data_copy['lower_bound'].astype(float).tolist(), mode='lines',
                line=dict(color='lightblue', width=0), fill='tonexty',
                fillcolor='rgba(31, 119, 180, 0.1)', name='Lower Bound', showlegend=False
            ), row=1, col=1)

        if 'sma' in data_copy.columns and not data_copy['sma'].isnull().all():
            fig.add_trace(go.Scatter(
                x=data_copy['Date'].tolist(), y=data_copy['sma'].astype(float).tolist(), mode='lines',
                line=dict(color='#1f77b4', width=1), name='SMA', showlegend=False
            ), row=1, col=1)

        # Row 2: RSI Plot
        if 'rsi' in data_copy.columns and not data_copy['rsi'].isnull().all():
            fig.add_trace(go.Scatter(
                x=data_copy['Date'].tolist(), y=data_copy['rsi'].astype(float).tolist(), mode='lines',
                line=dict(color='gray', width=2), name='RSI', showlegend=False
            ), row=2, col=1)
            fig.add_hline(y=70, line=dict(width=0.5), line_color='red', line_dash='solid', annotation_text="70",
                          annotation_position="top right", row=2, col=1, showlegend=False)
            fig.add_hline(y=30, line=dict(width=0.5), line_color='teal', line_dash='solid', annotation_text="30",
                          annotation_position="bottom right", row=2, col=1, showlegend=False)
        else:
            print("WARNING: RSI column is invalid or all NaN. Skipping RSI plot.")

        # Row 3: MACD Histogram Plot
        if 'MACD_Histo' in data_copy.columns and 'Color' in data_copy.columns and \
           not data_copy['MACD_Histo'].isnull().all():
            fig.add_trace(go.Bar(
                x=data_copy['Date'].tolist(), y=data_copy['MACD_Histo'].astype(float).tolist(),
                marker_color=data_copy['Color'], name='MACD', showlegend=False
            ), row=3, col=1)
            fig.add_hline(y=0, line_color='black', line_width=0.5, row=3, col=1, showlegend=False)
        else:
            print("WARNING: MACD_Histo or Color column is invalid or all NaN. Skipping MACD plot.")

        # Add title annotation
        ticker_name = data_copy['ticker_name'].iloc[0] if 'ticker_name' in data_copy.columns and not data_copy['ticker_name'].empty else "Stock Chart"
        fig.add_annotation(
            x=0, y=1, text=ticker_name, font=dict(size=20), showarrow=False,
            xanchor='left', yanchor='bottom', xref='x domain', yref='y domain',
            row=1, col=1
        )

        # Add price reference lines
        if len(data_copy) >= 2 and 'Close' in data_copy.columns and not data_copy['Close'].isnull().all():
            prev_close = float(data_copy.iloc[-2]['Close'])
            curr_close = float(data_copy.iloc[-1]['Close'])
            last_date = data_copy.iloc[-1]['Date']

            fig.add_hline(y=prev_close, line_color='black', line_width=0.3, row=1, col=1)
            fig.add_annotation(
                x=last_date, y=prev_close, text=f"{prev_close:.2f}",
                showarrow=False, xanchor="left", yanchor='top' if pchange >=0 else 'bottom',
                row=1, col=1
            )

            line_color = 'green' if pchange >= 0 else 'red'
            text_color = 'green' if pchange >= 0 else 'red'
            y_anchor = 'bottom' if pchange >= 0 else 'top'

            fig.add_hline(y=curr_close, line_color=line_color, line_width=0.3, row=1, col=1)
            fig.add_annotation(
                x=last_date, y=curr_close, text=f"{curr_close:.2f}",
                font=dict(color=text_color), showarrow=False,
                xanchor="left", yanchor=y_anchor,
                row=1, col=1
            )
        else:
            print("WARNING: Not enough valid 'Close' data for price reference lines after cleaning.")

        # Add portfolio holdings visualization if provided
        if portfolio_holdings and 'avg_price' in portfolio_holdings and 'quantity' in portfolio_holdings and \
           'Close' in data_copy.columns and not data_copy['Close'].isnull().all():
            avg_price = float(portfolio_holdings['avg_price'])
            quantity = float(portfolio_holdings['quantity'])
            
            fig.add_hline(
                y=avg_price, 
                line_color='blue', 
                line_width=1.5, 
                line_dash='dash',
                row=1, 
                col=1
            )
            
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
            
            current_price = float(data_copy.iloc[-1]['Close'])
            if current_price > avg_price:
                fig.add_shape(
                    type="rect", x0=data_copy.iloc[0]['Date'], x1=data_copy.iloc[-1]['Date'],
                    y0=avg_price, y1=current_price, fillcolor="rgba(0,255,0,0.1)",
                    line=dict(width=0), layer="below", row=1, col=1
                )
            elif current_price < avg_price:
                fig.add_shape(
                    type="rect", x0=data_copy.iloc[0]['Date'], x1=data_copy.iloc[-1]['Date'],
                    y0=current_price, y1=avg_price, fillcolor="rgba(255,0,0,0.1)",
                    line=dict(width=0), layer="below", row=1, col=1
                )
        else:
            print("WARNING: Portfolio holdings not plotted due to missing data or invalid 'Close' price after cleaning.")
        
        # Add vertical lines for signal crossings and trade signals for all subplots
        if 'z_cross' in data_copy.columns and not data_copy['z_cross'].isnull().all():
            z_cross_dates = data_copy[data_copy['z_cross'].isin([1, -1])]['Date'].tolist()
            for date_val in z_cross_dates:
                for r in [1, 2, 3]:
                    fig.add_vline(x=date_val, line_color='lightgray', line_width=0.3, row=r, col=1)
        else:
            print("WARNING: Z-cross signals not plotted due to missing or invalid 'z_cross' column.")

        if 'trade_signal' in data_copy.columns and not data_copy['trade_signal'].isnull().all():
            buy_signal_dates = data_copy[data_copy['trade_signal'] == 1]['Date'].tolist()
            sell_signal_dates = data_copy[data_copy['trade_signal'] == -1]['Date'].tolist()

            for date_val in buy_signal_dates:
                for r in [1, 2, 3]:
                    fig.add_vline(x=date_val, line_color='red', line_width=0.8, row=r, col=1)

            for date_val in sell_signal_dates:
                for r in [1, 2, 3]:
                    fig.add_vline(x=date_val, line_color='green', line_width=0.8, row=r, col=1)
        else:
            print("WARNING: Trade signals not plotted due to missing or invalid 'trade_signal' column.")

        # Add RSI and MACD subplot titles
        fig.add_annotation(x=0, y=1, text="RSI", font=dict(size=14), showarrow=False,
                           xanchor='left', xref='x domain', yref='y domain', row=2, col=1)
        fig.add_annotation(x=0, y=1, text="MACD", font=dict(size=14), showarrow=False,
                           xanchor='left', xref='x domain', yref='y domain', row=3, col=1)

        # --- RSI Divergence Detection and Plotting ---
        if 'rsi' in data_copy.columns and not data_copy['rsi'].isnull().all():
            # Ensure the index is DatetimeIndex for _detect_rsi_divergence
            original_index = data_copy.index
            try:
                # Create a temporary DataFrame with DatetimeIndex for divergence detection
                temp_data_for_div = data_copy.set_index(pd.to_datetime(data_copy['Date']))
                bullish_divs, bearish_divs = self._detect_rsi_divergence(temp_data_for_div, temp_data_for_div['rsi'])
            except Exception as e:
                print(f"WARNING: Could not convert 'Date' column to DatetimeIndex for divergence detection: {e}. Skipping divergence plots.")
                bullish_divs, bearish_divs = [], []
            finally:
                # Restore original index (important if data_copy is modified in place)
                data_copy.index = original_index


            # Plot Bullish Divergences
            for div_start_ts, div_end_ts in bullish_divs: # div_start_ts, div_end_ts are Timestamps
                # Find corresponding rows in data_copy using the original 'Date' string column
                start_row = data_copy[data_copy['Date'] == div_start_ts.strftime('%Y-%m-%d')]
                end_row = data_copy[data_copy['Date'] == div_end_ts.strftime('%Y-%m-%d')]

                if not start_row.empty and not end_row.empty:
                    price_start = start_row['Close'].iloc[0]
                    price_end = end_row['Close'].iloc[0]
                    rsi_start = start_row['rsi'].iloc[0]
                    rsi_end = end_row['rsi'].iloc[0]

                    # Ensure values are valid before plotting
                    if not np.isnan(price_start) and not np.isnan(price_end) and \
                       not np.isnan(rsi_start) and not np.isnan(rsi_end):
                        fig.add_trace(go.Scatter(x=[div_start_ts.strftime('%Y-%m-%d'), div_end_ts.strftime('%Y-%m-%d')],
                                                 y=[price_start, price_end],
                                                 mode='lines+markers', name='Bullish Div (Price)',
                                                 line=dict(color='black', dash='dash', width=2),
                                                 marker=dict(symbol='triangle-down', size=10, color='black'),
                                                 showlegend=False), row=1, col=1)
                        fig.add_trace(go.Scatter(x=[div_start_ts.strftime('%Y-%m-%d'), div_end_ts.strftime('%Y-%m-%d')],
                                                 y=[rsi_start, rsi_end],
                                                 mode='lines+markers', name='Bullish Div (RSI)',
                                                 line=dict(color='black', dash='dash', width=2),
                                                 marker=dict(symbol='triangle-up', size=10, color='black'),
                                                 showlegend=False), row=2, col=1)
                        # Add annotation for bullish divergence on price chart
                        fig.add_annotation(
                            x=div_end_ts.strftime('%Y-%m-%d'), y=price_end,
                            text="Bullish Div", showarrow=True, arrowhead=2, ax=20, ay=-40,
                            font=dict(color="lime", size=10), bgcolor="rgba(255,255,255,0.7)",
                            bordercolor="lime", borderwidth=1, borderpad=4,
                            row=1, col=1
                        )
                    else:
                        print(f"WARNING: Skipping bullish divergence plot due to NaN values at points: {div_start_ts.strftime('%Y-%m-%d')}, {div_end_ts.strftime('%Y-%m-%d')}")
                else:
                    print(f"WARNING: Skipping bullish divergence plot due to missing data for dates: {div_start_ts.strftime('%Y-%m-%d')}, {div_end_ts.strftime('%Y-%m-%d')}")


            # Plot Bearish Divergences
            for div_start_ts, div_end_ts in bearish_divs:
                start_row = data_copy[data_copy['Date'] == div_start_ts.strftime('%Y-%m-%d')]
                end_row = data_copy[data_copy['Date'] == div_end_ts.strftime('%Y-%m-%d')]

                if not start_row.empty and not end_row.empty:
                    price_start = start_row['Close'].iloc[0]
                    price_end = end_row['Close'].iloc[0]
                    rsi_start = start_row['rsi'].iloc[0]
                    rsi_end = end_row['rsi'].iloc[0]

                    if not np.isnan(price_start) and not np.isnan(price_end) and \
                       not np.isnan(rsi_start) and not np.isnan(rsi_end):
                        fig.add_trace(go.Scatter(x=[div_start_ts.strftime('%Y-%m-%d'), div_end_ts.strftime('%Y-%m-%d')],
                                                 y=[price_start, price_end],
                                                 mode='lines+markers', name='Bearish Div (Price)',
                                                 line=dict(color='black', dash='dash', width=2), # Changed color
                                                 marker=dict(symbol='triangle-up', size=10, color='black'), # Changed color
                                                 showlegend=False), row=1, col=1)
                        fig.add_trace(go.Scatter(x=[div_start_ts.strftime('%Y-%m-%d'), div_end_ts.strftime('%Y-%m-%d')],
                                                 y=[rsi_start, rsi_end],
                                                 mode='lines+markers', name='Bearish Div (RSI)',
                                                 line=dict(color='black', dash='dash', width=2), # Changed color
                                                 marker=dict(symbol='triangle-down', size=10, color='black'), # Changed color
                                                 showlegend=False), row=2, col=1)
                        # Add annotation for bearish divergence on price chart
                        fig.add_annotation(
                            x=div_end_ts.strftime('%Y-%m-%d'), y=price_end,
                            text="Bearish Div", showarrow=True, arrowhead=2, ax=20, ay=40,
                            font=dict(color="#CC0000", size=10), bgcolor="rgba(255,255,255,0.7)", # Changed color
                            bordercolor="#CC0000", borderwidth=1, borderpad=4, # Changed color
                            row=1, col=1
                        )
                    else:
                        print(f"WARNING: Skipping bearish divergence plot due to NaN values at points: {div_start_ts.strftime('%Y-%m-%d')}, {div_end_ts.strftime('%Y-%m-%d')}")
                else:
                    print(f"WARNING: Skipping bearish divergence plot due to missing data for dates: {div_start_ts.strftime('%Y-%m-%d')}, {div_end_ts.strftime('%Y-%m-%d')}")
        else:
            print("WARNING: RSI data not valid for divergence detection and plotting.")

        # Calculate dynamic y-axis range for price chart
        price_min = data_copy['Close'].min()
        price_max = data_copy['Close'].max()
        price_range_padding = (price_max - price_min) * 0.1 # 10% padding
        price_y_range = [price_min - price_range_padding, price_max + price_range_padding]

        # Update layout - with proper scaling for MACD and responsive sizing
        layout = {
            "autosize": True,
            "height": 600,
            "showlegend": False,
            "xaxis": {"rangeslider": {"visible": False}, "showgrid": False, "type": "date"}, # Explicitly set x-axis type
            "yaxis": {"type": "linear", "range": price_y_range, "showgrid": False}, # Explicitly set type and range
            "yaxis2": {"type": "linear", "range": [0, 100], "showticklabels": False, "showgrid": False}, # RSI range
            "yaxis3": {"type": "linear", "range": macd_range, "showticklabels": False, "showgrid": False}, # MACD range
            "plot_bgcolor": "white",
            "paper_bgcolor": "white",
            "hovermode": "x unified" # Improve hover experience
        }
        fig.update_layout(layout)

        # Convert to JSON for embedding in QWebEngineView
        return json.loads(fig.to_json())

    def create_current_figure(self, live_data: pd.DataFrame,
                              prev_close: Optional[float],
                             portfolio_holdings: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Create current day chart with volume

        Args:
            live_data: DataFrame with current day data
            prev_close: Previous day's close price
            portfolio_holdings: Optional dict with quantity and avg_price for portfolio visualization

        Returns:
            Plotly figure as JSON string for embedding in QWebEngineView.
        """
        if live_data.empty:
            # print("DEBUG: create_current_figure: Input live_data is empty.") # Removed debug
            fig = go.Figure()
            fig.update_layout(title="No current day data available")
            return json.loads(fig.to_json())

        live_data_copy = live_data.copy()

        # Ensure 'Datetime' column exists. If index is DatetimeIndex, create 'Datetime' column from it.
        if 'Datetime' not in live_data_copy.columns and isinstance(live_data_copy.index, pd.DatetimeIndex):
            live_data_copy['Datetime'] = live_data_copy.index.strftime('%Y-%m-%d %H:%M:%S')
        elif 'Datetime' not in live_data_copy.columns:
            print("WARNING: create_current_figure: 'Datetime' column missing and index is not DatetimeIndex. Using index as string.")
            live_data_copy['Datetime'] = live_data_copy.index.astype(str)


        # Convert all numeric columns to float to ensure proper plotting
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_cols:
            if col in live_data_copy.columns:
                live_data_copy[col] = pd.to_numeric(live_data_copy[col], errors='coerce')
            else:
                live_data_copy[col] = np.nan # Add missing columns with NaN

        # Fill NaNs for all plotted columns in live_data_copy
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in live_data_copy.columns:
                live_data_copy[col] = live_data_copy[col].fillna(0) # Fill leading NaNs with 0

        # Drop rows where critical OHLCV data is NaN after filling
        live_data_copy.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'], inplace=True)

        if live_data_copy.empty: # Handle case where all rows become NaN after cleaning
             # print("DEBUG: create_current_figure: No valid live data remaining after cleaning.") # Removed debug
             fig = go.Figure()
             fig.update_layout(title="No valid current day data available after cleaning")
             return json.loads(fig.to_json())

        # Create figure with secondary y-axis for volume
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Candlestick chart for price
        if all(col in live_data_copy.columns for col in ['Open', 'High', 'Low', 'Close']) and \
           not live_data_copy[['Open', 'High', 'Low', 'Close']].isnull().all().any():
            fig.add_trace(go.Candlestick(
                x=live_data_copy['Datetime'].tolist(),
                open=live_data_copy['Open'].astype(float).tolist(),
                close=live_data_copy['Close'].astype(float).tolist(),
                high=live_data_copy['High'].astype(float).tolist(),
                low=live_data_copy['Low'].astype(float).tolist(),
                name='Candlestick',
                showlegend=False
            ), secondary_y=False)
        else:
            print("WARNING: Not all OHLC data is valid for live candlestick plot. Plotting Close price as line.")
            if 'Close' in live_data_copy.columns and not live_data_copy['Close'].isnull().all():
                fig.add_trace(go.Scatter(
                    x=live_data_copy['Datetime'].tolist(), y=live_data_copy['Close'].astype(float).tolist(), mode='lines',
                    line=dict(color='blue'), name='Close Price', showlegend=False
                ), secondary_y=False)
            else:
                print("ERROR: Live 'Close' column also invalid for line plot.")

        # Bar chart for volume
        if 'Volume' in live_data_copy.columns and not live_data_copy['Volume'].isnull().all():
            fig.add_trace(go.Bar(
                x=live_data_copy['Datetime'].tolist(),
                y=live_data_copy['Volume'].astype(float).tolist(),
                name='Volume',
                marker_color='rgba(0,0,255,0.3)',
                showlegend=False
            ), secondary_y=True)
        else:
            print("WARNING: Live 'Volume' column is invalid or all NaN. Skipping volume plot.")

        # Previous close line
        if prev_close is not None:
            fig.add_hline(
                y = prev_close,
                line_color='black',
                line_width=0.5,
                annotation_text=f"Prev Close: {prev_close:.2f}",
                annotation_position="top right",
                secondary_y=False
            )

        # Add portfolio holdings visualization if provided
        if portfolio_holdings and 'avg_price' in portfolio_holdings and 'quantity' in portfolio_holdings and \
           'Close' in live_data_copy.columns and not live_data_copy['Close'].isnull().all():
            avg_price = float(portfolio_holdings['avg_price'])
            quantity = float(portfolio_holdings['quantity'])
            
            fig.add_hline(
                y=avg_price, 
                line_color='blue', 
                line_width=1.5, 
                line_dash='dash',
                secondary_y=False
            )
            
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
                ay=-40,
                secondary_y=False
            )
        else:
            print("WARNING: Portfolio holdings not plotted on live chart due to missing data or invalid 'Close' price.")
        
        # Calculate dynamic y-axis range for live price chart
        live_price_min = live_data_copy['Close'].min()
        live_price_max = live_data_copy['Close'].max()
        live_price_range_padding = (live_price_max - live_price_min) * 0.1 # 10% padding
        live_price_y_range = [live_price_min - live_price_range_padding, live_price_max + live_price_range_padding]

        # Update layout with responsive sizing and clean background
        layout = {
            "autosize": True,
            "height": 500, # Adjusted height for current day chart
            "title": "Current Day Trading",
            "showlegend": False,
            "xaxis": {"rangeslider": {"visible": False}, "showgrid": False, "title": "Time", "type": "date"}, # Explicitly set x-axis type
            "yaxis": {"showgrid": False, "title": "Price"},
            "yaxis2": {
                "showticklabels": False,
                "showgrid": False,
                "range": [0, live_data_copy['Volume'].max() * 10 if 'Volume' in live_data_copy.columns and not live_data_copy['Volume'].isnull().all() else 1000], # Increased range for volume clarity
                "title": "Volume"
            },
            "plot_bgcolor": "white",
            "paper_bgcolor": "white",
            "hovermode": "x unified"
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
            # print("DEBUG: create_portfolio_figure: Input holdings_data is empty.") # Removed debug
            fig = go.Figure()
            fig.update_layout(title="No portfolio holdings")
            return json.loads(fig.to_json())
        
        # Create figure
        fig = go.Figure()
        
        # Add pie chart for holdings allocation
        # Use 'ticker_name' for labels if available, otherwise fallback to 'ticker'
        labels = holdings_data['ticker_name'].tolist() if 'ticker_name' in holdings_data.columns else holdings_data['ticker'].tolist()
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
            # Example: fig.add_trace(go.Scatter(x=performance_data['Date'], y=performance_data['PortfolioValue']))
            pass
        
        # Update layout with responsive sizing and clean background
        layout = {
            "autosize": True,
            "height": 500, # Adjusted height for portfolio chart
            "title": "Portfolio Allocation",
            "showlegend": True,
            "legend": dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            "plot_bgcolor": "white",
            "paper_bgcolor": "white",
        }
        fig.update_layout(layout)
        
        # Convert to JSON for embedding in QWebEngineView
        return json.loads(fig.to_json())
