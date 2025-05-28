# PySide6 Stock Market Application Design

## Application Structure

The application will be structured with the following components:

1. **Main Window**: The primary application window containing all UI elements
2. **Data Module**: Handles data fetching, processing, and storage
3. **Chart Module**: Manages chart creation and visualization
4. **Ticker Management**: Interface for adding, editing, and removing tickers
5. **Transaction Management**: Interface for recording stock purchases and sales
6. **Portfolio Visualization**: Shows holdings on charts

## UI Layout Design

### Main Window
- Menu bar with File, Edit, View, and Help options
- Tab widget for switching between different views:
  - Market View (charts and data)
  - Portfolio View (holdings and performance)
  - Transaction History (list of all transactions)
- Status bar for displaying application status and notifications

### Market View Tab
- Left sidebar:
  - Ticker selection dropdown
  - Duration selection (slider or dropdown)
  - Refresh button
  - Key statistics display (current price, change, etc.)
- Main area:
  - Tab widget for switching between:
    - Historical chart (with technical indicators)
    - Current day chart
  - Chart will include portfolio holdings visualization

### Portfolio View Tab
- Summary section showing total portfolio value and performance
- Holdings table with columns:
  - Ticker
  - Quantity
  - Purchase Price
  - Current Price
  - Profit/Loss
  - Percentage Change
- Performance chart showing portfolio value over time

### Transaction History Tab
- Table view of all transactions with columns:
  - Date
  - Ticker
  - Transaction Type (Buy/Sell)
  - Quantity
  - Price
  - Total Value
- Filtering options by date range, ticker, and transaction type
- Add/Edit/Delete transaction buttons

### Ticker Management Dialog
- Table view of available tickers
- Add button to open dialog for adding new tickers
- Edit button to modify existing ticker details
- Delete button to remove tickers
- Import/Export buttons for ticker lists

### Transaction Entry Dialog
- Form with fields:
  - Date picker
  - Ticker selection (dropdown)
  - Transaction type (Buy/Sell)
  - Quantity
  - Price
  - Notes
- Calculate total value automatically
- Save and Cancel buttons

## Data Storage Design

### Tickers Data
- CSV file format for compatibility with existing code
- Fields: ticker symbol, display name
- Cached in memory for quick access

### Transactions Data
- SQLite database for efficient querying and persistence
- Tables:
  - Transactions (id, date, ticker, type, quantity, price, notes)
  - Portfolio (calculated from transactions)

### Market Data
- Fetched using yfinance as in the original code
- Cached for performance during session
- Technical indicators calculated as needed

## Portfolio Visualization Design

### On Historical Charts
- Horizontal lines showing average purchase price for holdings
- Annotations showing quantity held
- Color coding to indicate profit/loss positions

### On Current Day Charts
- Markers at purchase price points
- Tooltip showing purchase details on hover
- Visual indication of current position relative to day's movement

## Interaction Flow

1. Application starts and loads saved tickers and transactions
2. User selects ticker to view from dropdown
3. Application fetches and displays market data
4. User can:
   - Switch between historical and current views
   - Add/edit tickers through the management interface
   - Record new transactions
   - View portfolio performance
   - See holdings visualized on charts

## Technical Implementation Considerations

1. Use QThreads for background data fetching to keep UI responsive
2. Embed Plotly charts using QWebEngineView for interactive visualization
3. Implement data models for tables using Qt's model/view architecture
4. Use signals and slots for communication between components
5. Implement persistent settings using QSettings
