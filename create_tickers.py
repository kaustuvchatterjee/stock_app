"""
Create a sample tickers.csv file for the application
"""

import pandas as pd
import os

# Create sample tickers file if it doesn't exist
if not os.path.exists('tickers.csv'):
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
    
    # Create DataFrame
    df = pd.DataFrame({'ticker': tickers})
    
    # Save to CSV
    df.to_csv('tickers.csv', index=False)
    print("Created sample tickers.csv file")
else:
    print("tickers.csv already exists")
