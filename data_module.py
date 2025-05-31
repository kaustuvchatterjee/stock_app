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
                    # print("DEBUG: Ticker names missing or incomplete. Attempting to fetch from Yahoo Finance.")
                    self.ticker_names = []
                    for ticker in self.tickers:
                        try:
                            t = yf.Ticker(ticker)
                            info = t.info
                            # Prioritize 'shortName' or 'longName', fallback to ticker
                            name = info.get('shortName') or info.get('longName') or ticker
                            self.ticker_names.append(name)
                        except Exception as e:
                            print(f"WARNING: Could not fetch name for {ticker}: {e}")
                            self.ticker_names.append(ticker) # Fallback to ticker symbol
                    
                    # Update CSV with names
                    updated_df = pd.DataFrame({'ticker': self.tickers, 'name': self.ticker_names})
                    updated_df.to_csv(self.tickers_file, index=False)
                    print(f"Updated {self.tickers_file} with ticker names.")
            else:
                # Create sample tickers file if it doesn't exist
                tickers = [
                    '^NSEI',  # NIFTY 50
                    '^BSESN',  # SENSEX
                    'RELIANCE.NS',  # Reliance Industries
                    'TCS.NS',  # Tata Consultancy Services
                    'HDFCBANK.NS',  # HDFC Bank
                    'INFY.NS',  # Infosys
                    'AAPL',  # Apple
                    'MSFT',  # Microsoft
                    'GOOGL',  # Alphabet (Google)
                    'AMZN'   # Amazon
                ]
                # Fetch names for default tickers
                ticker_names = []
                for ticker in tickers:
                    try:
                        t = yf.Ticker(ticker)
                        info = t.info
                        name = info.get('shortName') or info.get('longName') or ticker
                        ticker_names.append(name)
                    except Exception as e:
                        print(f"WARNING: Could not fetch name for default ticker {ticker}: {e}")
                        ticker_names.append(ticker)
                
                df = pd.DataFrame({'ticker': tickers, 'name': ticker_names})
                df.to_csv(self.tickers_file, index=False)
                self.tickers = tickers
                self.ticker_names = ticker_names
                print(f"Created sample {self.tickers_file} file with names.")
        except Exception as e:
            print(f"ERROR: Failed to load or create tickers file: {e}")
            self.tickers = ['^NSEI'] # Fallback to a single ticker
            self.ticker_names = ['NIFTY 50']

    def _init_database(self) -> None:
        """Initialize the SQLite database for transactions"""
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS transactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    type TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    price REAL NOT NULL,
                    notes TEXT
                )
            """)
            conn.commit()
            conn.close()
            print(f"Database {self.db_file} initialized successfully.")
        except Exception as e:
            print(f"ERROR: Failed to initialize database: {e}")

    def get_tickers(self) -> Tuple[List[str], List[str]]:
        """Get list of available tickers and their names"""
        return self.tickers, self.ticker_names

    def add_ticker(self, ticker: str) -> bool:
        """Add a new ticker to the list and CSV"""
        if ticker in self.tickers:
            print(f"INFO: Ticker {ticker} already exists.")
            return False
        
        try:
            t = yf.Ticker(ticker)
            info = t.info
            name = info.get('shortName') or info.get('longName') or ticker
            
            self.tickers.append(ticker)
            self.ticker_names.append(name)
            
            df = pd.DataFrame({'ticker': self.tickers, 'name': self.ticker_names})
            df.to_csv(self.tickers_file, index=False)
            print(f"Added ticker {ticker} ({name}) to {self.tickers_file}")
            return True
        except Exception as e:
            print(f"ERROR: Failed to add ticker {ticker}: {e}")
            return False

    def remove_ticker(self, ticker: str) -> bool:
        """Remove a ticker from the list and CSV"""
        if ticker not in self.tickers:
            print(f"INFO: Ticker {ticker} not found.")
            return False
        
        try:
            idx = self.tickers.index(ticker)
            self.tickers.pop(idx)
            self.ticker_names.pop(idx)
            
            df = pd.DataFrame({'ticker': self.tickers, 'name': self.ticker_names})
            df.to_csv(self.tickers_file, index=False)
            print(f"Removed ticker {ticker} from {self.tickers_file}")
            return True
        except Exception as e:
            print(f"ERROR: Failed to remove ticker {ticker}: {e}")
            return False

    def get_ticker_data(self, ticker: str, duration: int) -> Tuple[pd.DataFrame, pd.DataFrame, str, int]:
        """
        Fetch historical and live data for a given ticker.
        
        Args:
            ticker: Ticker symbol (e.g., 'AAPL', '^NSEI')
            duration: Number of days for historical data (e.g., 180, 365)
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, str, int]:
                - Historical data (DataFrame)
                - Live (intraday) data (DataFrame)
                - Last updated timestamp (str)
                - Status code (1 for success, 0 for failure)
        """
        hist_data = pd.DataFrame()
        live_data = pd.DataFrame()
        last_updated = "N/A"
        status = 0

        try:
            # Explicitly cast ticker to string and wrap in a list for yfinance.download
            ticker_str = str(ticker)
            ticker_list = [ticker_str]
            # print(f"DEBUG: get_ticker_data: Inside, fetching for '{ticker_list[0]}' (type: {type(ticker_list[0])})")

            end_date = datetime.now(pytz.utc)
            start_date = end_date - timedelta(days=duration)

            # Fetch historical data
            hist_data_raw = yf.download(ticker_list, start=start_date, end=end_date, progress=False)
            # print(f"DEBUG: hist_data_raw after download (head):\n{hist_data_raw.head()}")
            
            # Fetch intraday data (for current day)
            live_data_raw = yf.download(ticker_list, period="1d", interval="1m", progress=False)
            # print(f"DEBUG: live_data_raw after download (head):\n{live_data_raw.head()}")

            # Extract data for the single ticker from the multi-level DataFrame
            # Use .xs to get the cross-section for the specific ticker, which flattens the columns
            if not hist_data_raw.empty and isinstance(hist_data_raw.columns, pd.MultiIndex):
                hist_data = hist_data_raw.xs(ticker_str, level=1, axis=1)
            else:
                hist_data = hist_data_raw # Already flat for a single ticker or empty

            if not live_data_raw.empty and isinstance(live_data_raw.columns, pd.MultiIndex):
                live_data = live_data_raw.xs(ticker_str, level=1, axis=1)
            else:
                live_data = live_data_raw # Already flat for a single ticker or empty

            if hist_data.empty and live_data.empty:
                print(f"WARNING: No data fetched for {ticker_list[0]} for duration {duration} days.")
                return pd.DataFrame(), pd.DataFrame(), "N/A", 0

            # Add ticker name to historical data ONLY IF hist_data is not empty
            if not hist_data.empty:
                ticker_name = ticker # Default to ticker symbol
                if ticker in self.tickers:
                    idx = self.tickers.index(ticker)
                    ticker_name = self.ticker_names[idx]
                hist_data['ticker_name'] = ticker_name

                # Calculate indicators for historical data
                hist_data = self._calculate_indicators(hist_data)
            else:
                print(f"WARNING: Historical data is empty for {ticker_list[0]}, skipping indicator calculation.")

            last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # print(f"DEBUG: Successfully fetched data for {ticker_list[0]}. Historical rows: {len(hist_data)}, Live rows: {len(live_data)}")
            status = 1

        except Exception as e:
            # Print the type of the exception and its message for better debugging
            print(f"ERROR: Failed to fetch data for {ticker_str}: {type(e).__name__}: {e}")
            status = 0
        
        return hist_data, live_data, last_updated, status

    def _calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate various technical indicators for the given DataFrame.
        Adds 'short', 'long', 'MACD', 'Signal', 'MACD_Histo', 'Candle',
        'Momentum', 'Dir', 'Color', 'neg', 'pos', 'z_cross', 'sma', 'stddev',
        'upper_bound', 'lower_bound', 'rsi', 'trade_signal' columns.
        """
        if data.empty or 'Close' not in data.columns:
            return data

        # Ensure 'Close' is numeric
        data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
        data.dropna(subset=['Close'], inplace=True)
        if data.empty:
            return data

        # Simple Moving Averages (for short and long term trends)
        data['short'] = data['Close'].rolling(window=12, min_periods=1).mean()
        data['long'] = data['Close'].rolling(window=26, min_periods=1).mean()

        # MACD
        exp1 = data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = exp1 - exp2
        data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
        data['MACD_Histo'] = data['MACD'] - data['Signal']

        # MACD Histogram Color
        data['Color'] = np.where(data['MACD_Histo'] < 0, '#EF5350', '#26A69A') # Red for negative, Green for positive

        # Candle (Price Change for easier visualization)
        data['Candle'] = data['Close'].diff().fillna(0)

        # Momentum Indicator (Rate of Change)
        data['Momentum'] = data['Close'].diff(periods=10).fillna(0) # 10-period momentum

        # Directional Movement (ADX components - simplified)
        data['Dir'] = np.where(data['Close'].diff() > 0, 1, -1) # 1 for up, -1 for down
        data['Dir'] = data['Dir'].replace(0, np.nan).ffill().fillna(0) # Handle no change, forward fill

        # Zero Cross for MACD Histogram
        data['neg'] = data['MACD_Histo'] < 0
        data['pos'] = data['MACD_Histo'] > 0
        data['z_cross'] = 0
        # Positive zero cross (MACD_Histo goes from negative to positive)
        data.loc[(data['neg'].shift(1) == True) & (data['pos'] == True), 'z_cross'] = 1
        # Negative zero cross (MACD_Histo goes from positive to negative)
        data.loc[(data['pos'].shift(1) == True) & (data['neg'] == True), 'z_cross'] = -1

        # Bollinger Bands
        window_bb = 20
        num_std_dev = 2
        data['sma'] = data['Close'].rolling(window=window_bb, min_periods=1).mean()
        data['stddev'] = data['Close'].rolling(window=window_bb, min_periods=1).std()
        data['upper_bound'] = data['sma'] + (data['stddev'] * num_std_dev)
        data['lower_bound'] = data['sma'] - (data['stddev'] * num_std_dev)

        # RSI - Now correctly calling the instance method
        data['rsi'] = self._calculate_rsi(data)

        # Simple Trade Signal (Example: Buy when MACD crosses Signal from below, Sell when from above)
        data['trade_signal'] = 0
        # Buy signal
        data.loc[(data['MACD'] > data['Signal']) & (data['MACD'].shift(1) <= data['Signal'].shift(1)), 'trade_signal'] = 1
        # Sell signal
        data.loc[(data['MACD'] < data['Signal']) & (data['MACD'].shift(1) >= data['Signal'].shift(1)), 'trade_signal'] = -1

        return data

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
            print("DEBUG: _calculate_rsi: 'Close' column not found or is all NaN.")
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

    def add_transaction(self, transaction: Dict[str, Any]) -> bool:
        """Add a new transaction to the database"""
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO transactions (date, ticker, type, quantity, price, notes)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (transaction['date'], transaction['ticker'], transaction['type'],
                  transaction['quantity'], transaction['price'], transaction['notes']))
            conn.commit()
            conn.close()
            print(f"Added transaction: {transaction}")
            return True
        except Exception as e:
            print(f"ERROR: Failed to add transaction: {e}")
            return False

    def get_transactions(self) -> List[Dict[str, Any]]:
        """Retrieve all transactions from the database"""
        transactions = []
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            cursor.execute("SELECT id, date, ticker, type, quantity, price, notes FROM transactions ORDER BY date DESC, id DESC")
            rows = cursor.fetchall()
            conn.close()

            for row in rows:
                transactions.append({
                    'id': row[0],
                    'date': row[1],
                    'ticker': row[2],
                    'type': row[3],
                    'quantity': row[4],
                    'price': row[5],
                    'notes': row[6]
                })
            print(f"Retrieved {len(transactions)} transactions.")
        except Exception as e:
            print(f"ERROR: Failed to retrieve transactions: {e}")
        return transactions

    def get_transaction_by_id(self, transaction_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve a single transaction by its ID"""
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            cursor.execute("SELECT id, date, ticker, type, quantity, price, notes FROM transactions WHERE id = ?", (transaction_id,))
            row = cursor.fetchone()
            conn.close()
            if row:
                return {
                    'id': row[0],
                    'date': row[1],
                    'ticker': row[2],
                    'type': row[3],
                    'quantity': row[4],
                    'price': row[5],
                    'notes': row[6]
                }
            return None
        except Exception as e:
            print(f"ERROR: Failed to retrieve transaction by ID {transaction_id}: {e}")
            return None

    def update_transaction(self, transaction_id: int, transaction: Dict[str, Any]) -> bool:
        """Update an existing transaction in the database"""
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE transactions
                SET date = ?, ticker = ?, type = ?, quantity = ?, price = ?, notes = ?
                WHERE id = ?
            """, (transaction['date'], transaction['ticker'], transaction['type'],
                  transaction['quantity'], transaction['price'], transaction['notes'],
                  transaction_id))
            conn.commit()
            conn.close()
            print(f"Updated transaction ID {transaction_id}: {transaction}")
            return True
        except Exception as e:
            print(f"ERROR: Failed to update transaction ID {transaction_id}: {e}")
            return False

    def delete_transaction(self, transaction_id: int) -> bool:
        """Delete a transaction from the database"""
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM transactions WHERE id = ?", (transaction_id,))
            conn.commit()
            conn.close()
            print(f"Deleted transaction ID {transaction_id}")
            return True
        except Exception as e:
            print(f"ERROR: Failed to delete transaction ID {transaction_id}: {e}")
            return False

    def get_portfolio_holdings(self) -> pd.DataFrame:
        """
        Calculate current portfolio holdings based on transactions.
        Returns a DataFrame with 'ticker', 'quantity', 'avg_price'.
        """
        transactions = self.get_transactions()
        if not transactions:
            return pd.DataFrame(columns=['ticker', 'quantity', 'avg_price'])

        holdings = {} # ticker -> {'quantity': X, 'cost_basis': Y}

        for trans in transactions:
            ticker = trans['ticker']
            quantity = trans['quantity']
            price = trans['price']
            trans_type = trans['type']

            if ticker not in holdings:
                holdings[ticker] = {'quantity': 0, 'cost_basis': 0}

            if trans_type == 'Buy':
                holdings[ticker]['quantity'] += quantity
                holdings[ticker]['cost_basis'] += (quantity * price)
            elif trans_type == 'Sell':
                # Handle selling more than held - for simplicity, just reduce quantity/cost
                # A more robust system would prevent shorting or handle it explicitly
                if holdings[ticker]['quantity'] >= quantity:
                    # Calculate average cost for the sold portion
                    avg_cost_per_share = holdings[ticker]['cost_basis'] / holdings[ticker]['quantity'] if holdings[ticker]['quantity'] > 0 else 0
                    holdings[ticker]['quantity'] -= quantity
                    holdings[ticker]['cost_basis'] -= (quantity * avg_cost_per_share)
                else:
                    # If selling more than held, zero out the holding
                    print(f"WARNING: Selling {quantity} of {ticker}, but only {holdings[ticker]['quantity']} held. Zeroing out holding.")
                    holdings[ticker]['quantity'] = 0
                    holdings[ticker]['cost_basis'] = 0

        # Filter out zero or negative holdings and calculate average price
        final_holdings = []
        for ticker, data in holdings.items():
            if data['quantity'] > 0:
                avg_price = data['cost_basis'] / data['quantity']
                final_holdings.append({'ticker': ticker, 'quantity': data['quantity'], 'avg_price': avg_price})

        holdings_df = pd.DataFrame(final_holdings)

        # Add current price and ticker name to holdings_df
        if not holdings_df.empty:
            current_prices = {}
            ticker_names_map = dict(zip(self.tickers, self.ticker_names))

            for ticker in holdings_df['ticker'].unique():
                # Pass the ticker as a list to get_ticker_data
                data, _, _, status = self.get_ticker_data(ticker, 1) 
                if status == 1 and not data.empty and 'Close' in data.columns and not data['Close'].isnull().all():
                    current_prices[ticker] = data.iloc[-1]['Close']
                else:
                    current_prices[ticker] = np.nan # Mark as NaN if price not available
            
            holdings_df['current_price'] = holdings_df['ticker'].map(current_prices)
            holdings_df['ticker_name'] = holdings_df['ticker'].map(ticker_names_map).fillna(holdings_df['ticker']) # Fallback to ticker if name not found

            # Drop rows where current_price is NaN, as we can't calculate value
            holdings_df.dropna(subset=['current_price'], inplace=True)

        return holdings_df

    def calculate_portfolio_value(self) -> Dict[str, float]:
        """
        Calculate the total current value, cost, profit/loss, and percentage change of the portfolio.
        """
        try:
            holdings = self.get_portfolio_holdings()
            if holdings.empty:
                return {'total_value': 0, 'total_cost': 0, 'profit_loss': 0, 'percent_change': 0}
            
            total_value = 0
            total_cost = 0
            
            # Ensure 'current_price' and 'avg_price' are numeric and not NaN
            holdings['current_price'] = pd.to_numeric(holdings['current_price'], errors='coerce')
            holdings['avg_price'] = pd.to_numeric(holdings['avg_price'], errors='coerce')
            holdings.dropna(subset=['current_price', 'avg_price'], inplace=True)

            if holdings.empty: # After dropping NaNs, if it's empty
                 return {'total_value': 0, 'total_cost': 0, 'profit_loss': 0, 'percent_change': 0}

            total_value = (holdings['quantity'] * holdings['current_price']).sum()
            total_cost = (holdings['quantity'] * holdings['avg_price']).sum()
            
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

