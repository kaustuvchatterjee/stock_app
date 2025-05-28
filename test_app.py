"""
Test script to validate the Stock Market Application
"""

import sys
import os
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import main application
from main import MainWindow

def test_application():
    """Run the application for testing"""
    app = QApplication(sys.argv)
    
    # Create main window
    window = MainWindow()
    window.show()
    
    # Set a timer to close the application after 30 seconds (for automated testing)
    # In real usage, comment out this line
    # QTimer.singleShot(30000, app.quit)
    
    # Start the application event loop
    sys.exit(app.exec())

if __name__ == "__main__":
    test_application()
