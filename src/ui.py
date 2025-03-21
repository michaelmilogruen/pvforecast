# -*- coding: utf-8 -*-
"""
Author: Michael Grün
Email: michaelgruen@hotmail.com
Description: Simple UI for PV power forecasting using the PVForecaster class.
Version: 1.0
Date: 2025-03-21
"""

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QComboBox, QPushButton, 
                            QSpinBox, QTableView, QHeaderView, QFileDialog,
                            QTabWidget, QMessageBox, QProgressBar)
from PyQt5.QtCore import Qt, QAbstractTableModel, QModelIndex
from PyQt5.QtGui import QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Import the PVForecaster class
from forecast import PVForecaster


class PandasModel(QAbstractTableModel):
    """Model for displaying pandas DataFrame in a QTableView"""
    
    def __init__(self, data):
        super().__init__()
        self._data = data

    def rowCount(self, parent=QModelIndex()):
        return self._data.shape[0]

    def columnCount(self, parent=QModelIndex()):
        return self._data.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid():
            if role == Qt.DisplayRole:
                value = self._data.iloc[index.row(), index.column()]
                # Format based on column type
                if isinstance(value, float):
                    return f"{value:.2f}"
                elif isinstance(value, pd.Timestamp):
                    return value.strftime('%Y-%m-%d %H:%M')
                else:
                    return str(value)
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return str(self._data.columns[section])
        if orientation == Qt.Vertical and role == Qt.DisplayRole:
            return str(self._data.index[section])
        return None


class MatplotlibCanvas(FigureCanvas):
    """Canvas for matplotlib figures"""
    
    def __init__(self, parent=None, width=10, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)


class PVForecastUI(QMainWindow):
    """Main UI window for PV power forecasting"""
    
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("PV Power Forecasting")
        self.setGeometry(100, 100, 1200, 800)
        
        # Initialize forecaster
        self.forecaster = None
        self.forecast_data = None
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create control panel
        control_panel = QWidget()
        control_layout = QHBoxLayout(control_panel)
        
        # Feature set selection
        feature_set_label = QLabel("Feature Set:")
        self.feature_set_combo = QComboBox()
        self.feature_set_combo.addItems(["inca", "station", "combined"])
        self.feature_set_combo.setCurrentText("station")  # Default
        control_layout.addWidget(feature_set_label)
        control_layout.addWidget(self.feature_set_combo)
        
        # Config ID selection
        config_id_label = QLabel("Config ID:")
        self.config_id_spin = QSpinBox()
        self.config_id_spin.setRange(1, 10)
        self.config_id_spin.setValue(1)  # Default
        control_layout.addWidget(config_id_label)
        control_layout.addWidget(self.config_id_spin)
        
        # Forecast hours
        hours_label = QLabel("Forecast Hours:")
        self.hours_spin = QSpinBox()
        self.hours_spin.setRange(24, 168)
        self.hours_spin.setValue(60)  # Default
        self.hours_spin.setSingleStep(12)
        control_layout.addWidget(hours_label)
        control_layout.addWidget(self.hours_spin)
        
        # Run forecast button
        self.run_button = QPushButton("Run Forecast")
        self.run_button.clicked.connect(self.run_forecast)
        control_layout.addWidget(self.run_button)
        
        # Export buttons
        self.export_csv_button = QPushButton("Export CSV")
        self.export_csv_button.clicked.connect(self.export_to_csv)
        self.export_csv_button.setEnabled(False)
        control_layout.addWidget(self.export_csv_button)
        
        self.export_json_button = QPushButton("Export JSON")
        self.export_json_button.clicked.connect(self.export_to_json)
        self.export_json_button.setEnabled(False)
        control_layout.addWidget(self.export_json_button)
        
        # Add control panel to main layout
        main_layout.addWidget(control_panel)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)
        
        # Create tab widget for plot and data
        self.tab_widget = QTabWidget()
        
        # Plot tab
        self.plot_tab = QWidget()
        plot_layout = QVBoxLayout(self.plot_tab)
        self.canvas = MatplotlibCanvas(self.plot_tab, width=12, height=6)
        plot_layout.addWidget(self.canvas)
        
        # Data tab
        self.data_tab = QWidget()
        data_layout = QVBoxLayout(self.data_tab)
        self.table_view = QTableView()
        self.table_view.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        data_layout.addWidget(self.table_view)
        
        # Add tabs to tab widget
        self.tab_widget.addTab(self.plot_tab, "Forecast Plot")
        self.tab_widget.addTab(self.data_tab, "Forecast Data")
        
        # Add tab widget to main layout
        main_layout.addWidget(self.tab_widget)
        
        # Status bar
        self.statusBar().showMessage("Ready")
    
    def run_forecast(self):
        """Run the forecast with the selected parameters"""
        try:
            # Show progress bar
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(10)
            self.statusBar().showMessage("Initializing forecaster...")
            
            # Get parameters
            feature_set = self.feature_set_combo.currentText()
            config_id = self.config_id_spin.value()
            hours = self.hours_spin.value()
            
            # Initialize forecaster
            self.forecaster = PVForecaster(feature_set=feature_set, config_id=config_id)
            
            self.progress_bar.setValue(30)
            self.statusBar().showMessage("Fetching weather data...")
            
            # Make predictions
            self.forecast_data = self.forecaster.predict(hours=hours)
            
            self.progress_bar.setValue(70)
            self.statusBar().showMessage("Generating plot...")
            
            # Plot forecast
            self.canvas.axes.clear()
            fig = self.forecaster.plot_forecast(self.forecast_data, show=False)
            self.canvas.fig.clear()
            self.canvas.fig = fig
            self.canvas.draw()
            
            self.progress_bar.setValue(90)
            self.statusBar().showMessage("Updating data table...")
            
            # Update table view
            model = PandasModel(self.forecast_data)
            self.table_view.setModel(model)
            
            # Enable export buttons
            self.export_csv_button.setEnabled(True)
            self.export_json_button.setEnabled(True)
            
            self.progress_bar.setValue(100)
            self.statusBar().showMessage(f"Forecast completed: {feature_set} config {config_id}, {hours} hours")
            
            # Hide progress bar after a delay
            self.progress_bar.setVisible(False)
            
        except Exception as e:
            self.progress_bar.setVisible(False)
            QMessageBox.critical(self, "Error", f"Error running forecast: {str(e)}")
            self.statusBar().showMessage(f"Error: {str(e)}")
    
    def export_to_csv(self):
        """Export forecast data to CSV"""
        if self.forecast_data is None:
            return
        
        try:
            # Get file path from user
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save CSV File", "data/forecast_data.csv", "CSV Files (*.csv)")
            
            if file_path:
                # Export to CSV
                self.forecaster.export_to_csv(self.forecast_data, filename=file_path)
                self.statusBar().showMessage(f"Data exported to {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error exporting to CSV: {str(e)}")
            self.statusBar().showMessage(f"Error: {str(e)}")
    
    def export_to_json(self):
        """Export forecast data to JSON"""
        if self.forecast_data is None:
            return
        
        try:
            # Get file path from user
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save JSON File", "data/forecast_data.json", "JSON Files (*.json)")
            
            if file_path:
                # Export to JSON
                self.forecaster.export_to_json(self.forecast_data, filename=file_path)
                self.statusBar().showMessage(f"Data exported to {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error exporting to JSON: {str(e)}")
            self.statusBar().showMessage(f"Error: {str(e)}")


def main():
    """Main function to run the UI"""
    app = QApplication(sys.argv)
    window = PVForecastUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
