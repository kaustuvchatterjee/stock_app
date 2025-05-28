# Stock Market Application User Guide

## Overview

This PySide6-based Stock Market Application allows you to:

1. Track stock market data with interactive charts
2. Manage your portfolio of stocks and mutual funds
3. Record purchase and sale transactions
4. Visualize your holdings directly on price charts

## Features

### Market View
- Select from multiple tickers or add your own
- View historical charts with technical indicators (MACD, RSI, Bollinger Bands)
- See current day's price movement and volume
- Portfolio holdings are visualized directly on charts

### Portfolio Management
- Track your current holdings across multiple stocks
- View portfolio allocation and performance
- See profit/loss for each position

### Transaction Management
- Record buy and sell transactions
- View transaction history
- Calculate average purchase price and position size

## Installation

1. Ensure you have Python 3.8+ installed
2. Install required packages:
   ```
   pip install PySide6 pandas numpy yfinance plotly
   ```
3. Run the application:
   ```
   python main.py
   ```

## Usage Instructions

### Managing Tickers
1. Go to File → Manage Tickers
2. Use the Add button to add new ticker symbols
3. Use Edit to modify existing tickers
4. Use Delete to remove tickers

### Viewing Stock Data
1. Select a ticker from the dropdown in the Market View tab
2. Adjust the duration using the spinner
3. Click Refresh to update the data
4. Switch between Historical and Current tabs to view different charts

### Recording Transactions
1. Go to Transaction → Add Transaction
2. Select the date, ticker, and transaction type (Buy/Sell)
3. Enter quantity and price
4. Add optional notes
5. Click Save

### Viewing Your Portfolio
1. Go to the Portfolio tab
2. See your current holdings and their performance
3. The pie chart shows your portfolio allocation

### Viewing Holdings on Charts
- When viewing a stock you own, your holdings will be displayed as:
  - A horizontal line showing your average purchase price
  - An annotation showing quantity and purchase price
  - Color-coded areas indicating profit/loss zones

## Troubleshooting

- If charts don't load, ensure you have an active internet connection
- If a ticker doesn't work, verify the symbol is correct
- For any data issues, try clicking the Refresh button

## Technical Notes

- Stock data is fetched using the yfinance package
- Charts are created using Plotly and embedded in the application
- Transaction data is stored in a local SQLite database
- Ticker information is stored in a CSV file
