"""
Main application for Stock Market GUI
Implements the PySide6 GUI for stock market data visualization and portfolio management
"""

from PySide6.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, 
                              QHBoxLayout, QLabel, QComboBox, QSlider, QPushButton, QSpinBox,
                              QTableWidget, QTableWidgetItem, QHeaderView, QDialog, QFormLayout,
                              QLineEdit, QDateEdit, QDoubleSpinBox, QRadioButton, QGroupBox,
                              QMessageBox, QSplitter, QFrame, QStatusBar, QMenu, QMenuBar)
from PySide6.QtCore import Qt, QDate, Signal, Slot, QThread, QUrl
from PySide6.QtGui import QFont, QColor, QIcon, QAction

from PySide6.QtWebEngineWidgets import QWebEngineView

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

# Import custom modules
from data_module import DataModule
from chart_module import ChartModule


class StockDataThread(QThread):
    """Thread for fetching stock data in background"""
    data_ready = Signal(object, object, str, int)
    
    def __init__(self, data_module, ticker, duration):
        super().__init__()
        self.data_module = data_module
        self.ticker = ticker
        self.duration = duration
        self.running = True
    
    def run(self):
        if self.running:
            data, live_data, last_updated, status = self.data_module.get_ticker_data(self.ticker, self.duration)
            if self.running:  # Check again before emitting signal
                self.data_ready.emit(data, live_data, last_updated, status)
    
    def stop(self):
        """Stop the thread safely"""
        self.running = False
        self.wait()  # Wait for the thread to finish


class TickerDialog(QDialog):
    """Dialog for adding or editing tickers"""
    def __init__(self, parent=None, ticker=""):
        super().__init__(parent)
        self.setWindowTitle("Add Ticker" if not ticker else "Edit Ticker")
        self.setMinimumWidth(300)
        
        # Create layout
        layout = QFormLayout()
        
        # Create widgets
        self.ticker_input = QLineEdit(ticker)
        self.ticker_input.setPlaceholderText("Enter ticker symbol (e.g., AAPL)")
        
        # Add widgets to layout
        layout.addRow("Ticker Symbol:", self.ticker_input)
        
        # Create buttons
        button_layout = QHBoxLayout()
        self.ok_button = QPushButton("OK")
        self.cancel_button = QPushButton("Cancel")
        
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)
        
        layout.addRow("", button_layout)
        
        # Set layout
        self.setLayout(layout)
        
        # Connect signals
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)
    
    def get_ticker(self):
        """Get the entered ticker symbol"""
        return self.ticker_input.text().strip().upper()


class TransactionDialog(QDialog):
    """Dialog for adding or editing transactions"""
    def __init__(self, parent=None, tickers=None, ticker_names=None, transaction=None):
        super().__init__(parent)
        self.setWindowTitle("Add Transaction" if not transaction else "Edit Transaction")
        self.setMinimumWidth(400)
        
        # Store tickers
        self.tickers = tickers or []
        self.ticker_names = ticker_names or []
        
        # Create layout
        layout = QFormLayout()
        
        # Create widgets
        self.date_edit = QDateEdit(QDate.currentDate())
        self.date_edit.setCalendarPopup(True)
        
        self.ticker_combo = QComboBox()
        for i, name in enumerate(self.ticker_names):
            self.ticker_combo.addItem(name, self.tickers[i])
        
        self.type_group = QGroupBox("Transaction Type")
        type_layout = QHBoxLayout()
        self.buy_radio = QRadioButton("Buy")
        self.sell_radio = QRadioButton("Sell")
        self.buy_radio.setChecked(True)
        type_layout.addWidget(self.buy_radio)
        type_layout.addWidget(self.sell_radio)
        self.type_group.setLayout(type_layout)
        
        self.quantity_spin = QDoubleSpinBox()
        self.quantity_spin.setRange(0.001, 1000000)
        self.quantity_spin.setDecimals(3)
        self.quantity_spin.setValue(1)
        
        self.price_spin = QDoubleSpinBox()
        self.price_spin.setRange(0.01, 1000000)
        self.price_spin.setDecimals(2)
        self.price_spin.setValue(100)
        
        self.notes_input = QLineEdit()
        self.notes_input.setPlaceholderText("Optional notes")
        
        self.total_label = QLabel("₹100.00")
        
        # Add widgets to layout
        layout.addRow("Date:", self.date_edit)
        layout.addRow("Ticker:", self.ticker_combo)
        layout.addRow("", self.type_group)
        layout.addRow("Quantity:", self.quantity_spin)
        layout.addRow("Price:", self.price_spin)
        layout.addRow("Notes:", self.notes_input)
        layout.addRow("Total Value:", self.total_label)
        
        # Create buttons
        button_layout = QHBoxLayout()
        self.ok_button = QPushButton("Save")
        self.cancel_button = QPushButton("Cancel")
        
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)
        
        layout.addRow("", button_layout)
        
        # Set layout
        self.setLayout(layout)
        
        # Connect signals
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)
        self.quantity_spin.valueChanged.connect(self.update_total)
        self.price_spin.valueChanged.connect(self.update_total)
        
        # Fill with transaction data if provided
        if transaction is not None:
            self.date_edit.setDate(QDate.fromString(transaction['date'], "yyyy-MM-dd"))
            
            # Set ticker
            ticker_idx = self.tickers.index(transaction['ticker']) if transaction['ticker'] in self.tickers else 0
            self.ticker_combo.setCurrentIndex(ticker_idx)
            
            # Set type
            if transaction['type'] == 'Buy':
                self.buy_radio.setChecked(True)
            else:
                self.sell_radio.setChecked(True)
            
            self.quantity_spin.setValue(transaction['quantity'])
            self.price_spin.setValue(transaction['price'])
            self.notes_input.setText(transaction['notes'])
            
            self.update_total()
    
    def update_total(self):
        """Update the total value label"""
        quantity = self.quantity_spin.value()
        price = self.price_spin.value()
        total = quantity * price
        self.total_label.setText(f"₹{total:.2f}")
    
    def get_transaction(self):
        """Get the transaction data"""
        return {
            'date': self.date_edit.date().toString("yyyy-MM-dd"),
            'ticker': self.ticker_combo.currentData(),
            'type': 'Buy' if self.buy_radio.isChecked() else 'Sell',
            'quantity': self.quantity_spin.value(),
            'price': self.price_spin.value(),
            'notes': self.notes_input.text()
        }


class TickerManagementDialog(QDialog):
    """Dialog for managing tickers"""
    def __init__(self, parent=None, data_module=None):
        super().__init__(parent)
        self.setWindowTitle("Manage Tickers")
        self.setMinimumSize(500, 400)
        
        self.data_module = data_module
        
        # Create layout
        layout = QVBoxLayout()
        
        # Create table
        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Ticker", "Name"])
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        
        # Create buttons
        button_layout = QHBoxLayout()
        self.add_button = QPushButton("Add")
        self.edit_button = QPushButton("Edit")
        self.delete_button = QPushButton("Delete")
        self.close_button = QPushButton("Close")
        
        button_layout.addWidget(self.add_button)
        button_layout.addWidget(self.edit_button)
        button_layout.addWidget(self.delete_button)
        button_layout.addStretch()
        button_layout.addWidget(self.close_button)
        
        # Add widgets to layout
        layout.addWidget(self.table)
        layout.addLayout(button_layout)
        
        # Set layout
        self.setLayout(layout)
        
        # Connect signals
        self.add_button.clicked.connect(self.add_ticker)
        self.edit_button.clicked.connect(self.edit_ticker)
        self.delete_button.clicked.connect(self.delete_ticker)
        self.close_button.clicked.connect(self.accept)
        
        # Load tickers
        self.load_tickers()
    
    def load_tickers(self):
        """Load tickers from data module"""
        if self.data_module:
            tickers, ticker_names = self.data_module.get_tickers()
            
            self.table.setRowCount(len(tickers))
            for i, (ticker, name) in enumerate(zip(tickers, ticker_names)):
                self.table.setItem(i, 0, QTableWidgetItem(ticker))
                self.table.setItem(i, 1, QTableWidgetItem(name))
    
    def add_ticker(self):
        """Add a new ticker"""
        dialog = TickerDialog(self)
        if dialog.exec():
            ticker = dialog.get_ticker()
            if ticker:
                if self.data_module.add_ticker(ticker):
                    self.load_tickers()
                else:
                    QMessageBox.warning(self, "Error", f"Failed to add ticker {ticker}")
    
    def edit_ticker(self):
        """Edit selected ticker"""
        selected = self.table.selectedItems()
        if selected and selected[0].column() == 0:
            row = selected[0].row()
            current_ticker = self.table.item(row, 0).text()
            
            dialog = TickerDialog(self, current_ticker)
            if dialog.exec():
                new_ticker = dialog.get_ticker()
                if new_ticker and new_ticker != current_ticker:
                    # Remove old ticker and add new one
                    if self.data_module.remove_ticker(current_ticker) and self.data_module.add_ticker(new_ticker):
                        self.load_tickers()
                    else:
                        QMessageBox.warning(self, "Error", f"Failed to update ticker {current_ticker} to {new_ticker}")
        else:
            QMessageBox.information(self, "Select Ticker", "Please select a ticker to edit")
    
    def delete_ticker(self):
        """Delete selected ticker"""
        selected = self.table.selectedItems()
        if selected and selected[0].column() == 0:
            row = selected[0].row()
            ticker = self.table.item(row, 0).text()
            
            reply = QMessageBox.question(self, "Confirm Delete", 
                                        f"Are you sure you want to delete ticker {ticker}?",
                                        QMessageBox.Yes | QMessageBox.No)
            
            if reply == QMessageBox.Yes:
                if self.data_module.remove_ticker(ticker):
                    self.load_tickers()
                else:
                    QMessageBox.warning(self, "Error", f"Failed to delete ticker {ticker}")
        else:
            QMessageBox.information(self, "Select Ticker", "Please select a ticker to delete")


class MainWindow(QMainWindow):
    """Main application window"""
    def __init__(self):
        super().__init__()
        
        # Initialize modules
        self.data_module = DataModule()
        self.chart_module = ChartModule()
        
        # Initialize tickers and ticker names before setting up tabs
        self.tickers, self.ticker_names = [], [] # Initialize as empty lists
        
        # Set window properties
        self.setWindowTitle("Algotrade")
        self.setMinimumSize(800, 600)
        
        # Create central widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # Create main layout
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        self.main_layout.addWidget(self.tab_widget)
        
        # Create tabs
        self.market_tab = QWidget()
        self.portfolio_tab = QWidget()
        self.transactions_tab = QWidget()
        
        self.tab_widget.addTab(self.market_tab, "Market View")
        self.tab_widget.addTab(self.portfolio_tab, "Portfolio")
        self.tab_widget.addTab(self.transactions_tab, "Transactions")
        
        # Setup tabs (market tab must be setup before load_tickers to initialize ticker_combo)
        self.setup_market_tab()
        self.setup_portfolio_tab()
        self.setup_transactions_tab()
        
        # Now load tickers, as ticker_combo is initialized
        self.load_tickers() 
        
        # Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        # Create menu bar
        self.setup_menu_bar()
        
        # Initialize data
        self.current_ticker_idx = 0
        self.current_duration = 180
        self.historical_data = pd.DataFrame()
        self.live_data = pd.DataFrame()
        self.last_updated = "N/A"
        self.data_thread = None
        
        # Load initial data
        self.fetch_data()
    
    def closeEvent(self, event):
        """Handle window close event to properly clean up threads"""
        # Stop any running data thread
        if self.data_thread is not None and self.data_thread.isRunning():
            self.data_thread.stop()
        
        # Accept the close event
        event.accept()
    
    def setup_menu_bar(self):
        """Setup the menu bar"""
        menu_bar = QMenuBar()
        self.setMenuBar(menu_bar)
        
        # File menu
        file_menu = menu_bar.addMenu("File")
        
        manage_tickers_action = QAction("Manage Tickers", self)
        manage_tickers_action.triggered.connect(self.show_ticker_management)
        file_menu.addAction(manage_tickers_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Transaction menu
        transaction_menu = menu_bar.addMenu("Transaction")
        
        add_transaction_action = QAction("Add Transaction", self)
        add_transaction_action.triggered.connect(self.add_transaction)
        transaction_menu.addAction(add_transaction_action)
    
    def setup_market_tab(self):
        """Setup the market view tab"""
        layout = QVBoxLayout(self.market_tab)
        
        # Create top section with controls
        top_layout = QHBoxLayout()
        
        # Left side controls
        controls_layout = QVBoxLayout()
        
        # Ticker selection
        ticker_layout = QHBoxLayout()
        ticker_label = QLabel("Ticker:")
        self.ticker_combo = QComboBox() # self.ticker_combo is initialized here
        self.ticker_combo.currentIndexChanged.connect(self.on_ticker_changed)
        ticker_layout.addWidget(ticker_label)
        ticker_layout.addWidget(self.ticker_combo)
        controls_layout.addLayout(ticker_layout)
        
        # Duration selection
        duration_layout = QHBoxLayout()
        duration_label = QLabel("Duration (days):")
        self.duration_spin = QSpinBox()
        self.duration_spin.setRange(30, 730)
        self.duration_spin.setValue(180)
        self.duration_spin.setSingleStep(30)
        self.duration_spin.valueChanged.connect(self.on_duration_changed)
        duration_layout.addWidget(duration_label)
        duration_layout.addWidget(self.duration_spin)
        controls_layout.addLayout(duration_layout)
        
        # Refresh button
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.fetch_data)
        controls_layout.addWidget(self.refresh_button)
        
        top_layout.addLayout(controls_layout)
        
        # Right side stats
        stats_layout = QVBoxLayout()
        
        # Current price and change
        self.price_label = QLabel("N/A")
        self.price_label.setAlignment(Qt.AlignCenter)
        font = QFont()
        font.setPointSize(24)
        font.setBold(True)
        self.price_label.setFont(font)
        
        # Stats grid
        stats_grid = QHBoxLayout()
        self.prev_close_label = QLabel("Prev Close: N/A")
        self.high_label = QLabel("High: N/A")
        self.open_label = QLabel("Open: N/A")
        self.low_label = QLabel("Low: N/A")
        
        stats_left = QVBoxLayout()
        stats_left.addWidget(self.prev_close_label)
        stats_left.addWidget(self.high_label)
        
        stats_right = QVBoxLayout()
        stats_right.addWidget(self.open_label)
        stats_right.addWidget(self.low_label)
        
        stats_grid.addLayout(stats_left)
        stats_grid.addLayout(stats_right)
        
        self.last_updated_label = QLabel("Last Updated: N/A")
        self.last_updated_label.setAlignment(Qt.AlignRight)
        
        stats_layout.addWidget(self.price_label)
        stats_layout.addLayout(stats_grid)
        stats_layout.addWidget(self.last_updated_label)
        
        top_layout.addLayout(stats_layout)
        
        layout.addLayout(top_layout)
        
        # Create chart tabs
        chart_tabs = QTabWidget()
        
        # Historical chart tab
        self.historical_tab = QWidget()
        historical_layout = QVBoxLayout(self.historical_tab)
        self.historical_web_view = QWebEngineView()
        historical_layout.addWidget(self.historical_web_view)
        
        # Current chart tab
        self.current_tab = QWidget()
        current_layout = QVBoxLayout(self.current_tab)
        self.current_web_view = QWebEngineView()
        current_layout.addWidget(self.current_web_view)
        
        chart_tabs.addTab(self.historical_tab, "Historical")
        chart_tabs.addTab(self.current_tab, "Current")
        
        layout.addWidget(chart_tabs)
    
    def setup_portfolio_tab(self):
        """Setup the portfolio tab"""
        layout = QVBoxLayout(self.portfolio_tab)
        
        # Summary section
        summary_frame = QFrame()
        summary_frame.setFrameShape(QFrame.StyledPanel)
        summary_layout = QHBoxLayout(summary_frame)
        
        # Portfolio value
        value_layout = QVBoxLayout()
        value_label = QLabel("Portfolio Value")
        value_label.setAlignment(Qt.AlignCenter)
        self.portfolio_value_label = QLabel("₹0.00")
        self.portfolio_value_label.setAlignment(Qt.AlignCenter)
        font = QFont()
        font.setPointSize(24)
        font.setBold(True)
        self.portfolio_value_label.setFont(font)
        value_layout.addWidget(value_label)
        value_layout.addWidget(self.portfolio_value_label)
        
        # Profit/Loss
        pl_layout = QVBoxLayout()
        pl_label = QLabel("Profit/Loss")
        pl_label.setAlignment(Qt.AlignCenter)
        self.pl_value_label = QLabel("₹0.00 (0.00%)")
        self.pl_value_label.setAlignment(Qt.AlignCenter)
        font = QFont()
        font.setPointSize(24)
        font.setBold(True)
        self.pl_value_label.setFont(font)
        pl_layout.addWidget(pl_label)
        pl_layout.addWidget(self.pl_value_label)
        
        summary_layout.addLayout(value_layout)
        summary_layout.addLayout(pl_layout)
        
        layout.addWidget(summary_frame)
        
        # Holdings table
        holdings_label = QLabel("Current Holdings")
        holdings_label.setAlignment(Qt.AlignCenter)
        font = QFont()
        font.setPointSize(14)
        holdings_label.setFont(font)
        
        self.holdings_table = QTableWidget()
        self.holdings_table.setColumnCount(6)
        self.holdings_table.setHorizontalHeaderLabels(["Ticker", "Name", "Quantity", "Avg Price", "Current Price", "Profit/Loss"])
        self.holdings_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        layout.addWidget(holdings_label)
        layout.addWidget(self.holdings_table)
        
        # Portfolio chart
        chart_label = QLabel("Portfolio Allocation")
        chart_label.setAlignment(Qt.AlignCenter)
        font = QFont()
        font.setPointSize(14)
        chart_label.setFont(font)
        
        self.portfolio_web_view = QWebEngineView()
        
        layout.addWidget(chart_label)
        layout.addWidget(self.portfolio_web_view)
    
    def setup_transactions_tab(self):
        """Setup the transactions tab"""
        layout = QVBoxLayout(self.transactions_tab)
        
        # Buttons for managing transactions
        button_layout = QHBoxLayout()
        self.add_trans_button = QPushButton("Add Transaction")
        self.edit_trans_button = QPushButton("Edit Transaction")
        self.delete_trans_button = QPushButton("Delete Transaction")
        
        button_layout.addWidget(self.add_trans_button)
        button_layout.addWidget(self.edit_trans_button)
        button_layout.addWidget(self.delete_trans_button)
        button_layout.addStretch()
        
        # Transactions table
        self.transactions_table = QTableWidget()
        self.transactions_table.setColumnCount(7)
        self.transactions_table.setHorizontalHeaderLabels(["ID", "Date", "Ticker", "Type", "Quantity", "Price", "Total"])
        self.transactions_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.transactions_table.setColumnHidden(0, True)  # Hide ID column
        
        layout.addLayout(button_layout)
        layout.addWidget(self.transactions_table)
        
        # Connect signals
        self.add_trans_button.clicked.connect(self.add_transaction)
        self.edit_trans_button.clicked.connect(self.edit_transaction)
        self.delete_trans_button.clicked.connect(self.delete_transaction)
        
        # Load transactions
        self.load_transactions()
    
    def load_tickers(self):
        """Load tickers from data module"""
        self.tickers, self.ticker_names = self.data_module.get_tickers()
        
        # Update ticker combo
        self.ticker_combo.clear()
        for name in self.ticker_names:
            self.ticker_combo.addItem(name)
    
    def on_ticker_changed(self, index):
        """Handle ticker selection change"""
        if index >= 0 and index < len(self.tickers):
            self.current_ticker_idx = index
            self.fetch_data()
    
    def on_duration_changed(self, value):
        """Handle duration change"""
        self.current_duration = value
        self.fetch_data()
    
    def fetch_data(self):
        """Fetch data for current ticker"""
        if not self.tickers:
            return
        
        self.status_bar.showMessage(f"Fetching data for {self.ticker_names[self.current_ticker_idx]}...")
        self.refresh_button.setEnabled(False)
        
        # Stop any existing thread
        if self.data_thread is not None and self.data_thread.isRunning():
            self.data_thread.stop()
        
        # Start background thread for data fetching
        self.data_thread = StockDataThread(
            self.data_module, 
            self.tickers[self.current_ticker_idx], 
            self.current_duration
        )
        self.data_thread.data_ready.connect(self.on_data_ready)
        self.data_thread.start()
    
    def on_data_ready(self, data, live_data, last_updated, status):
        """Handle data fetching completion"""
        self.refresh_button.setEnabled(True)
        
        if status == 1:
            self.historical_data = data
            self.live_data = live_data
            self.last_updated = last_updated
            
            # Calculate percentage change
            pchange = 100 * (data.iloc[-1]['Close'] - data.iloc[-2]['Close']) / data.iloc[-2]['Close']
            
            # Update UI
            self.update_stats(data, pchange)
            self.update_charts(data, live_data, pchange)
            
            self.status_bar.showMessage(f"Data updated for {self.ticker_names[self.current_ticker_idx]}")
        else:
            self.status_bar.showMessage(f"Error fetching data for {self.ticker_names[self.current_ticker_idx]}")
    
    def update_stats(self, data, pchange):
        """Update statistics display"""
        # Update price and change
        price = data.iloc[-1]['Close']
        if pchange > 0:
            self.price_label.setText(f"{price:.2f} ▲ ({pchange:.2f}%)")
            self.price_label.setStyleSheet("color: green;")
        else:
            self.price_label.setText(f"{price:.2f} ▼ ({pchange:.2f}%)")
            self.price_label.setStyleSheet("color: red;")
        
        # Update other stats
        self.prev_close_label.setText(f"Prev Close: {data.iloc[-2]['Close']:.2f}")
        self.high_label.setText(f"High: {data.iloc[-1]['High']:.2f}")
        self.open_label.setText(f"Open: {data.iloc[-1]['Open']:.2f}")
        self.low_label.setText(f"Low: {data.iloc[-1]['Low']:.2f}")
        
        self.last_updated_label.setText(f"Last Updated: {self.last_updated}")
    
    def update_charts(self, data, live_data, pchange):
        """Update chart displays"""
        # Get portfolio holdings for current ticker
        holdings = self.data_module.get_portfolio_holdings()
        current_ticker = self.tickers[self.current_ticker_idx]
        
        portfolio_holdings = None
        if not holdings.empty and current_ticker in holdings['ticker'].values:
            ticker_holding = holdings[holdings['ticker'] == current_ticker].iloc[0]
            portfolio_holdings = {
                'quantity': ticker_holding['quantity'],
                'avg_price': ticker_holding['avg_price']
            }
        
        # Update historical chart
        historical_fig = self.chart_module.create_historical_figure(data, pchange, portfolio_holdings)
        historical_html = self.create_plotly_html(historical_fig)
        self.historical_web_view.setHtml(historical_html)
        
        # Update current chart
        prev_close = data.iloc[-2]['Close']
        current_fig = self.chart_module.create_current_figure(live_data, prev_close, portfolio_holdings)
        current_html = self.create_plotly_html(current_fig)
        self.current_web_view.setHtml(current_html)
        
        # Update portfolio tab
        self.update_portfolio_tab()
    
    def create_plotly_html(self, fig_json):
        """Create HTML with embedded Plotly chart"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <script src="https://cdn.plot.ly/plotly-2.24.1.min.js"></script>
        </head>
        <body>
            <div id="chart" style="width: 100%; height: 100%;"></div>
            <script>
                var figure = {json.dumps(fig_json)};
                Plotly.newPlot('chart', figure.data, figure.layout);
                
                window.onresize = function() {{
                    Plotly.Plots.resize('chart');
                }};
            </script>
        </body>
        </html>
        """
        return html
    
    def update_portfolio_tab(self):
        """Update portfolio tab with current holdings"""
        # Get holdings
        holdings = self.data_module.get_portfolio_holdings()
        
        # Clear table
        self.holdings_table.setRowCount(0)
        
        if not holdings.empty:
            # Add current prices and ticker names to holdings DataFrame
            holdings['current_price'] = 0.0
            holdings['profit_loss'] = 0.0
            holdings['profit_loss_pct'] = 0.0
            holdings['ticker_name'] = "" # Add ticker_name column
            
            for i, row in holdings.iterrows():
                ticker = row['ticker']
                data, _, _, status = self.data_module.get_ticker_data(ticker, 1)
                
                if status == 1 and not data.empty:
                    current_price = data.iloc[-1]['Close']
                    holdings.at[i, 'current_price'] = current_price
                    
                    # Calculate profit/loss
                    avg_price = row['avg_price']
                    quantity = row['quantity']
                    profit_loss = (current_price - avg_price) * quantity
                    profit_loss_pct = (current_price - avg_price) / avg_price * 100
                    
                    holdings.at[i, 'profit_loss'] = profit_loss
                    holdings.at[i, 'profit_loss_pct'] = profit_loss_pct

                # Get ticker name and assign to the holdings DataFrame
                ticker_name = ticker
                if ticker in self.tickers:
                    idx = self.tickers.index(ticker)
                    ticker_name = self.ticker_names[idx]
                holdings.at[i, 'ticker_name'] = ticker_name
            
            # Update table
            self.holdings_table.setRowCount(len(holdings))
            
            for i, row in holdings.iterrows():
                # Use row['ticker_name'] directly as it's now in the DataFrame
                ticker_name_display = row['ticker_name']
                
                # Add to table
                self.holdings_table.setItem(i, 0, QTableWidgetItem(row['ticker']))
                self.holdings_table.setItem(i, 1, QTableWidgetItem(ticker_name_display))
                self.holdings_table.setItem(i, 2, QTableWidgetItem(f"{row['quantity']:.3f}"))
                self.holdings_table.setItem(i, 3, QTableWidgetItem(f"{row['avg_price']:.2f}"))
                self.holdings_table.setItem(i, 4, QTableWidgetItem(f"{row['current_price']:.2f}"))
                
                pl_item = QTableWidgetItem(f"{row['profit_loss']:.2f} ({row['profit_loss_pct']:.2f}%)")
                if row['profit_loss'] > 0:
                    pl_item.setForeground(QColor("green"))
                elif row['profit_loss'] < 0:
                    pl_item.setForeground(QColor("red"))
                self.holdings_table.setItem(i, 5, pl_item)
            
            # Update portfolio summary
            portfolio_value = holdings['quantity'] * holdings['current_price']
            total_value = portfolio_value.sum()
            total_cost = (holdings['quantity'] * holdings['avg_price']).sum()
            profit_loss = total_value - total_cost
            profit_loss_pct = (profit_loss / total_cost) * 100 if total_cost > 0 else 0
            
            self.portfolio_value_label.setText(f"₹{total_value:.2f}")
            
            if profit_loss >= 0:
                self.pl_value_label.setText(f"₹{profit_loss:.2f} ({profit_loss_pct:.2f}%)")
                self.pl_value_label.setStyleSheet("color: green;")
            else:
                self.pl_value_label.setText(f"₹{profit_loss:.2f} ({profit_loss_pct:.2f}%)")
                self.pl_value_label.setStyleSheet("color: red;")
            
            # Update portfolio chart
            portfolio_fig = self.chart_module.create_portfolio_figure(holdings)
            portfolio_html = self.create_plotly_html(portfolio_fig)
            self.portfolio_web_view.setHtml(portfolio_html)
        else:
            # No holdings
            self.portfolio_value_label.setText("₹0.00")
            self.pl_value_label.setText("₹0.00 (0.00%)")
            self.pl_value_label.setStyleSheet("")
            
            # Empty chart
            empty_fig = {"data": [], "layout": {"title": "No portfolio holdings"}}
            portfolio_html = self.create_plotly_html(empty_fig)
            self.portfolio_web_view.setHtml(portfolio_html)
    
    def load_transactions(self):
        """Load transactions into table"""
        transactions = self.data_module.get_transactions()
        
        self.transactions_table.setRowCount(0)

        if not transactions.empty:
            self.transactions_table.setRowCount(len(transactions))
            
            for i, row in transactions.iterrows():
                self.transactions_table.setItem(i, 0, QTableWidgetItem(str(row['id'])))
                self.transactions_table.setItem(i, 1, QTableWidgetItem(row['date']))
                
                # Get ticker name
                ticker = row['ticker']
                ticker_name = ticker
                # Ensure self.tickers is populated before checking
                if hasattr(self, 'tickers') and ticker in self.tickers:
                    idx = self.tickers.index(ticker)
                    ticker_name = self.ticker_names[idx]
                
                self.transactions_table.setItem(i, 2, QTableWidgetItem(f"{ticker} ({ticker_name})"))
                
                type_item = QTableWidgetItem(row['type'])
                if row['type'] == 'Buy':
                    type_item.setForeground(QColor("green"))
                else:
                    type_item.setForeground(QColor("red"))
                self.transactions_table.setItem(i, 3, type_item)
                
                self.transactions_table.setItem(i, 4, QTableWidgetItem(f"{row['quantity']:.3f}"))
                self.transactions_table.setItem(i, 5, QTableWidgetItem(f"{row['price']:.2f}"))
                
                total = row['quantity'] * row['price']
                self.transactions_table.setItem(i, 6, QTableWidgetItem(f"{total:.2f}"))
    
    def show_ticker_management(self):
        """Show ticker management dialog"""
        dialog = TickerManagementDialog(self, self.data_module)
        if dialog.exec():
            self.load_tickers()
            self.fetch_data()
    
    def add_transaction(self):
        """Add a new transaction"""
        dialog = TransactionDialog(self, self.tickers, self.ticker_names)
        if dialog.exec():
            transaction = dialog.get_transaction()
            
            if self.data_module.add_transaction(
                transaction['date'],
                transaction['ticker'],
                transaction['type'],
                transaction['quantity'],
                transaction['price'],
                transaction['notes']
            ):
                self.load_transactions()
                self.update_portfolio_tab()
                
                # Update charts if current ticker matches transaction
                if transaction['ticker'] == self.tickers[self.current_ticker_idx]:
                    self.fetch_data()
            else:
                QMessageBox.warning(self, "Error", "Failed to add transaction")
    
    def edit_transaction(self):
        """Edit selected transaction"""
        selected = self.transactions_table.selectedItems()
        if selected:
            row = selected[0].row()
            transaction_id = int(self.transactions_table.item(row, 0).text())
            
            # Get transaction data
            transactions = self.data_module.get_transactions()
            transaction = transactions[transactions['id'] == transaction_id].iloc[0].to_dict()
            
            dialog = TransactionDialog(self, self.tickers, self.ticker_names, transaction)
            if dialog.exec():
                # Delete old transaction and add new one
                # This is a simplification - in a real app, you'd update the existing record
                # To properly update, you'd need a method in DataModule to update a transaction by ID
                # For now, I'm assuming a delete and re-add for simplicity as per original code's comment
                # However, the original code doesn't actually delete, just reloads.
                # Let's add a proper delete method to data_module and use it.
                if self.data_module.delete_transaction(transaction_id) and \
                   self.data_module.add_transaction(
                    dialog.get_transaction()['date'],
                    dialog.get_transaction()['ticker'],
                    dialog.get_transaction()['type'],
                    dialog.get_transaction()['quantity'],
                    dialog.get_transaction()['price'],
                    dialog.get_transaction()['notes']
                ):
                    self.load_transactions()
                    self.update_portfolio_tab()
                    self.fetch_data()
                else:
                    QMessageBox.warning(self, "Error", "Failed to update transaction")
        else:
            QMessageBox.information(self, "Select Transaction", "Please select a transaction to edit")
    
    def delete_transaction(self):
        """Delete selected transaction"""
        selected = self.transactions_table.selectedItems()
        if selected:
            row = selected[0].row()
            transaction_id = int(self.transactions_table.item(row, 0).text())
            
            reply = QMessageBox.question(self, "Confirm Delete", 
                                        "Are you sure you want to delete this transaction?",
                                        QMessageBox.Yes | QMessageBox.No)
            
            if reply == QMessageBox.Yes:
                if self.data_module.delete_transaction(transaction_id): # Use the new delete method
                    self.load_transactions()
                    self.update_portfolio_tab()
                    self.fetch_data()
                else:
                    QMessageBox.warning(self, "Error", "Failed to delete transaction")
        else:
            QMessageBox.information(self, "Select Transaction", "Please select a transaction to edit")


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
