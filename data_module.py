"""
Data Module for Stock Market Application
Handles data fetching, processing, and storage
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import pytz
import os
import sqlite3
from typing import Tuple, List, Dict, Optional, Any


class DataModule:
    def __init__(self, tickers_file='tickers.csv', db_file='transactions.db'):
        """Initialize the data module with file paths for tickers and transactions"""
        self.tickers_file = tickers_file
        self.db_file = db_file
        self.tickers = []
        self.ticker_names = []
        self.cached_data = {}
        
        # Create or load tickers file
        self._load_tickers()
        
        # Initialize database
        self._init_database()
    
    def _load_tickers(self) -> None:
        """Load tickers from CSV file or create default if not exists"""
        try:
            if os.path.exists(self.tickers_file):
                df = pd.read_csv(self.tickers_file)
                self.tickers = df['ticker'].to_list()
                self.ticker_names = df['name'].to_list() if 'name' in df.columns else []
                
                # If ticker names not available, fetch them
                if not self.ticker_names or len(self.ticker_names) != len(self.tickers):
                    self.ticker_names = []
                    for ticker in self.tickers:
                        try:
                            t = yf.Ticker(ticker)
                            self.ticker_names.append(t.info['shortName'])
                        except Exception:
                            self.ticker_names.append(ticker)
                    
                    # Save updated ticker names
                    self._save_tickers()
            else:
                # Create default tickers file
                self.tickers = ['^NSEI']  # Default to NIFTY 50
                t = yf.Ticker('^NSEI')
                self.ticker_names = [t.info['shortName']]
                self._save_tickers()
                
        except Exception as e:
            print(f"Error loading tickers: {e}")
            self.tickers = ['^NSEI']
            self.ticker_names = ['NIFTY 50']
            self._save_tickers()
    
    def _save_tickers(self) -> None:
        """Save tickers to CSV file"""
        df = pd.DataFrame({
            'ticker': self.tickers,
            'name': self.ticker_names
        })
        df.to_csv(self.tickers_file, index=False)
    
    def _init_database(self) -> None:
        """Initialize SQLite database for transactions"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        # Create transactions table if not exists
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            ticker TEXT NOT NULL,
            type TEXT NOT NULL,
            quantity REAL NOT NULL,
            price REAL NOT NULL,
            notes TEXT
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def get_tickers(self) -> Tuple[List[str], List[str]]:
        """Get list of tickers and their display names"""
        return self.tickers, self.ticker_names
    
    def add_ticker(self, ticker: str) -> bool:
        """Add a new ticker to the list"""
        try:
            # Check if ticker already exists
            if ticker in self.tickers:
                return False
            
            # Validate ticker by fetching data
            t = yf.Ticker(ticker)
            ticker_name = t.info['shortName']
            
            # Add to lists
            self.tickers.append(ticker)
            self.ticker_names.append(ticker_name)
            
            # Save to file
            self._save_tickers()
            return True
        except Exception as e:
            print(f"Error adding ticker {ticker}: {e}")
            return False
    
    def remove_ticker(self, ticker: str) -> bool:
        """Remove a ticker from the list"""
        try:
            if ticker in self.tickers:
                idx = self.tickers.index(ticker)
                self.tickers.pop(idx)
                self.ticker_names.pop(idx)
                self._save_tickers()
                return True
            return False
        except Exception as e:
            print(f"Error removing ticker {ticker}: {e}")
            return False
    
    def get_ticker_data(self, ticker: str, duration: int = 180) -> Tuple[pd.DataFrame, pd.DataFrame, str, int]:
        """
        Fetch historical and live data for a ticker
        
        Args:
            ticker: Ticker symbol
            duration: Number of days of historical data to fetch
            
        Returns:
            Tuple of (historical_data, live_data, last_updated, status)
        """
        data = pd.DataFrame()
        live_data = pd.DataFrame()
        last_updated = 'N/A'
        status = 0
        
        try:
            # Check cache first
            cache_key = f"{ticker}_{duration}"
            if cache_key in self.cached_data:
                cache_time, cached_result = self.cached_data[cache_key]
                # Use cache if less than 5 minutes old
                if (datetime.now() - cache_time).total_seconds() < 300:
                    return cached_result
            
            t = yf.Ticker(ticker)
            tz = pytz.timezone(t.info['exchangeTimezoneName'])
            end_date = datetime.today()
            end_date = end_date.astimezone(tz=tz)
            start_date = end_date + timedelta(days=-duration)
            
            data = yf.download(ticker, start=start_date, end=end_date)
            if data.empty:
                raise ValueError(f"No historical data downloaded for {ticker}")
            
            data.columns = data.columns.droplevel(1)
            
            # Ensure index is timezone-naive to avoid Plotly issues
            if data.index.tz is not None:
                data.index = data.index.tz_localize(None)
            
            quote_type = t.info['quoteType']
            if quote_type in ['INDEX', 'EQUITY']:
                live_data = t.history(period='1d', interval='1m')
                live_data.reset_index(inplace=True)
                
                # Ensure Datetime column is timezone-naive
                if pd.api.types.is_datetime64_any_dtype(live_data['Datetime']):
                    if live_data['Datetime'].dt.tz is not None:
                        live_data['Datetime'] = live_data['Datetime'].dt.tz_localize(None)
                else:
                    live_data['Datetime'] = pd.to_datetime(live_data['Datetime']).dt.tz_localize(None)
                
                last_updated = datetime.strftime(live_data.iloc[-1]['Datetime'], '%d %b %Y %H:%M')
            elif quote_type == 'MUTUALFUND':
                live_data = t.history(period='1mo', interval='1d')
                live_data.reset_index(inplace=True)
                
                # Rename Date to Datetime for consistency
                if 'Date' in live_data.columns:
                    live_data.rename(columns={'Date': 'Datetime'}, inplace=True)
                
                # Ensure Datetime column is timezone-naive
                if pd.api.types.is_datetime64_any_dtype(live_data['Datetime']):
                    if live_data['Datetime'].dt.tz is not None:
                        live_data['Datetime'] = live_data['Datetime'].dt.tz_localize(None)
                else:
                    live_data['Datetime'] = pd.to_datetime(live_data['Datetime']).dt.tz_localize(None)
                
                last_updated = datetime.strftime(live_data.iloc[-1]['Datetime'], '%d %b %Y %H:%M')
            else:
                last_updated = "N/A"
            
            status = 1
            
            # Process data with technical indicators
            if not data.empty:
                data = self.get_macd(data)
                
                # Ensure Date column is properly formatted for Plotly
                if 'Date' in data.columns:
                    data['Date'] = pd.to_datetime(data['Date'])
                
                # Add ticker name for display
                data['ticker_name'] = self.ticker_names[self.tickers.index(ticker)] if ticker in self.tickers else ticker
            
            # Cache the result
            result = (data, live_data, last_updated, status)
            self.cached_data[cache_key] = (datetime.now(), result)

            return result
        except Exception as error:
            print(f"Error fetching data for {ticker}: {error}")
            return pd.DataFrame(), pd.DataFrame(), "Error", 0
    
    def get_rsi(self, data: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate RSI for the given data"""
        change = data["Close"].diff()
        change_up = change.copy()
        change_down = change.copy()
        
        change_up[change_up < 0] = 0
        change_down[change_down > 0] = 0
        
        avg_up = change_up.rolling(window).mean()
        avg_down = change_down.rolling(window).mean().abs()
        
        rsi = 100 * avg_up / (avg_up + avg_down)
        return rsi
    
    def get_macd(self, data: pd.DataFrame, short_window: int = 12, long_window: int = 26, 
                signal_window: int = 9, bollinger_window: int = 20) -> pd.DataFrame:
        """Calculate MACD and other technical indicators for the given data"""
        # Copy data to avoid modifying original
        df = data.copy()
        
        # MACD
        df['short'] = df['Close'].ewm(span=short_window).mean()
        df['long'] = df['Close'].ewm(span=long_window).mean()
        df['MACD'] = df['short'] - df['long']
        df['Signal'] = df['MACD'].rolling(signal_window).mean()
        df['MACD_Histo'] = df['MACD'] - df['Signal']
        df['Candle'] = df['Close'] - df['Open']
        df['Momentum'] = df['Candle'].rolling(7).mean()
        df['Dir'] = df['MACD_Histo'].diff()
        
        # Color coding for MACD histogram
        df['Color'] = 'white'
        for i in range(len(df)):
            if df.iloc[i]['MACD_Histo'] > 0:
                if df.iloc[i]['Dir'] > 0:
                    df.at[df.index[i], 'Color'] = '#008080'
                else:
                    df.at[df.index[i], 'Color'] = '#b2d8d8'
            elif df.iloc[i]['MACD_Histo'] < 0:
                if df.iloc[i]['Dir'] > 0:
                    df.at[df.index[i], 'Color'] = '#ef7753'
                else:
                    df.at[df.index[i], 'Color'] = '#ec4242'
            else:
                df.at[df.index[i], 'Color'] = 'white'
        
        # Zero Crossing of MACD Histogram
        df['neg'] = (df['MACD'] > df['Signal']) & (df['MACD'].shift(1) <= df['Signal'].shift(1))
        df['pos'] = (df['MACD'] < df['Signal']) & (df['MACD'].shift(1) >= df['Signal'].shift(1))
        df['z_cross'] = np.where(df['neg'], -1, np.where(df['pos'], 1, 0))
        
        # Bollinger Bands
        df['sma'] = df['Close'].rolling(window=bollinger_window).mean()
        df['stddev'] = df['Close'].rolling(window=bollinger_window).std()
        df['upper_bound'] = df['sma'] + (df['stddev'] * 2)
        df['lower_bound'] = df['sma'] - (df['stddev'] * 2)
        
        # RSI
        df['rsi'] = self.get_rsi(df)
        
        # Trade Signals
        cf = 0.05
        df['trade_signal'] = 0
        for i in range(len(df)):
            if (df.iloc[i]['Close'] > df.iloc[i]['upper_bound']) or \
               ((df.iloc[i]['Close'] > df.iloc[i]['upper_bound'] - cf * (df.iloc[i]['upper_bound'] - df.iloc[i]['lower_bound'])) and \
                (df.iloc[i]['rsi'] > 75)):
                df.at[df.index[i], 'trade_signal'] = 1
            elif (df.iloc[i]['Close'] < df.iloc[i]['lower_bound'] + \
                 (cf * (df.iloc[i]['upper_bound'] - df.iloc[i]['lower_bound']))):
                df.at[df.index[i], 'trade_signal'] = -1
        
        # Reset index and ensure Date column is properly formatted
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'Date'}, inplace=True)
        
        # Ensure Date column is datetime type and timezone-naive
        if pd.api.types.is_datetime64_any_dtype(df['Date']):
            if df['Date'].dt.tz is not None:
                df['Date'] = df['Date'].dt.tz_localize(None)
        else:
            df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)

        return df
    
    def add_transaction(self, date: str, ticker: str, trans_type: str, 
                       quantity: float, price: float, notes: str = "") -> bool:
        """Add a new transaction to the database"""
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            
            cursor.execute('''
            INSERT INTO transactions (date, ticker, type, quantity, price, notes)
            VALUES (?, ?, ?, ?, ?, ?)
            ''', (date, ticker, trans_type, quantity, price, notes))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error adding transaction: {e}")
            return False
    
    def get_transactions(self, ticker: Optional[str] = None) -> pd.DataFrame:
        """Get all transactions, optionally filtered by ticker"""
        try:
            conn = sqlite3.connect(self.db_file)
            
            if ticker:
                query = "SELECT * FROM transactions WHERE ticker = ? ORDER BY date"
                df = pd.read_sql_query(query, conn, params=(ticker,))
            else:
                query = "SELECT * FROM transactions ORDER BY date"
                df = pd.read_sql_query(query, conn)
            
            conn.close()
            return df
        except Exception as e:
            print(f"Error getting transactions: {e}")
            return pd.DataFrame()
    
    def get_portfolio_holdings(self) -> pd.DataFrame:
        """Calculate current portfolio holdings from transactions"""
        try:
            transactions = self.get_transactions()
            if transactions.empty:
                return pd.DataFrame(columns=['ticker', 'quantity', 'avg_price'])
            
            # Group by ticker and calculate net position
            holdings = []
            for ticker in transactions['ticker'].unique():
                ticker_txns = transactions[transactions['ticker'] == ticker]
                
                # Calculate total quantity and average price
                buy_txns = ticker_txns[ticker_txns['type'] == 'Buy']
                sell_txns = ticker_txns[ticker_txns['type'] == 'Sell']
                
                total_bought = buy_txns['quantity'].sum()
                total_sold = sell_txns['quantity'].sum()
                net_quantity = total_bought - total_sold
                
                if net_quantity > 0:
                    # Calculate average purchase price (weighted average)
                    total_cost = (buy_txns['quantity'] * buy_txns['price']).sum()
                    avg_price = total_cost / total_bought
                    
                    holdings.append({
                        'ticker': ticker,
                        'quantity': net_quantity,
                        'avg_price': avg_price
                    })
            
            return pd.DataFrame(holdings)
        except Exception as e:
            print(f"Error calculating holdings: {e}")
            return pd.DataFrame(columns=['ticker', 'quantity', 'avg_price'])
    
    def get_portfolio_value(self) -> Dict[str, Any]:
        """Calculate current portfolio value and performance"""
        try:
            holdings = self.get_portfolio_holdings()
            if holdings.empty:
                return {'total_value': 0, 'total_cost': 0, 'profit_loss': 0, 'percent_change': 0}
            
            total_value = 0
            total_cost = 0
            
            for _, row in holdings.iterrows():
                ticker = row['ticker']
                quantity = row['quantity']
                avg_price = row['avg_price']
                
                # Get current price
                data, _, _, status = self.get_ticker_data(ticker, 1)
                if status == 1 and not data.empty:
                    current_price = data.iloc[-1]['Close']
                    
                    position_value = quantity * current_price
                    position_cost = quantity * avg_price
                    
                    total_value += position_value
                    total_cost += position_cost
            
            profit_loss = total_value - total_cost
            percent_change = (profit_loss / total_cost) * 100 if total_cost > 0 else 0
            
            return {
                'total_value': total_value,
                'total_cost': total_cost,
                'profit_loss': profit_loss,
                'percent_change': percent_change
            }
        except Exception as e:
            print(f"Error calculating portfolio value: {e}")
            return {'total_value': 0, 'total_cost': 0, 'profit_loss': 0, 'percent_change': 0}
