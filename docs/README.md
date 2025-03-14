# 🔮 PV Forecast

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange.svg)](https://tensorflow.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Advanced photovoltaic power generation forecasting system utilizing deep learning and weather data integration.

📚 **[Read our Project Paper](Forecasting_PV_Power_LSTM_Simulated_Data_Grün_Gressl_Rinnhofer.pdf)** - Comprehensive documentation of our forecasting methodology and results

## 🏗️ Architecture

```mermaid
graph TD
    A[Weather Data] --> C[Data Processing]
    B[PV Measurements] --> C
    C --> D[LSTM Models]
    D --> E[Power Forecast]
    E --> F[Visualization]
```

## 📁 Project Structure

```
pvforecast/
├── 📝 src/                  # Source code
│   ├── 🔮 forecast.py       # Main forecasting logic
│   ├── 🧠 lstm.py          # LSTM model implementations
│   └── 🖥️ ui.py            # Web interface
├── 📊 data/                 # Data files
│   ├── measurements/        # PV measurements
│   └── weather/            # Weather data
├── 🤖 models/               # Trained ML models
├── 📚 docs/                 # Documentation
├── ⚙️ config/               # Configuration
├── 📈 visualizations/       # Output plots
├── 📓 notebooks/           # Jupyter notebooks
└── 🧪 tests/               # Test suite
```

## 🌟 Key Features

- **Advanced ML Models**: LSTM-based architecture for time series forecasting
- **Multi-Source Data Integration**: Combines weather forecasts with historical PV data
- **Real-time Processing**: Continuous data ingestion and prediction pipeline
- **Interactive Visualization**: Web-based dashboard for forecast monitoring
- **Scalable Architecture**: Modular design for easy extension

## 🔄 Data Processing Pipeline

The project implements a sophisticated data processing pipeline that combines multiple data sources. For detailed implementation, see our [Data Processing Documentation](data_processing.md).


### Data Sources
- **INCA Weather Data**: Hourly meteorological data including:
  - Global radiation (W/m²)
  - Temperature (°C)
  - Relative humidity (%)
  - Wind components (m/s)

- **Local Weather Station**: 10-minute interval measurements with quality control:
  - Global radiation
  - Temperature
  - Wind speed
  - Precipitation
  - Pressure
  - Quality flags for critical measurements

- **PV System Data**: Production data including:
  - Energy yield (Wh)
  - Interval energy
  - Power output (W)

### Processing Steps
1. **Data Loading**: Efficient chunk-based CSV reading with customizable parameters
2. **Quality Control**: 
   - Filtering based on quality flags for weather station data
   - Handling of missing values and invalid measurements
3. **Temporal Alignment**: 
   - Resampling all data to 15-minute intervals
   - Appropriate aggregation methods for different parameters
4. **Feature Engineering**:
   - Clear sky radiation calculation using pvlib
   - Clear sky index computation
   - Time-based features (hour, day of year)
   - Solar position features (hour/day sine/cosine)
5. **Data Integration**:
   - Merging of all data sources on timestamp
   - Handling of missing values
   - Final export to optimized parquet format

## 💻 Technical Stack

- **Backend**
  - 🐍 Python 3.8+
  - 🧠 TensorFlow/Keras
  - 📊 NumPy/Pandas/pvlib
  - 🚀 FastAPI
  - 📦 PyArrow (Parquet)

- **Frontend**
  - 📈 D3.js
  - ⚛️ React
  - 📱 Responsive Design

## ⚡ Quick Start

1. **Environment Setup**
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   .\venv\Scripts\activate  # Windows
   
   # Install dependencies
   pip install -r requirements.txt
   ```

2. **Configuration**
   ```bash
   # Copy example config
   cp config/config.example.json config/config.json
   
   # Edit configuration
   nano config/config.json
   ```

3. **Run Application**
   ```bash
   # Start the forecasting system
   python src/forecast.py
   
   # Launch web interface
   python src/ui.py
   ```

## 📊 Performance Metrics

| Model | MAE (kW) | RMSE (kW) | Forecast Horizon |
|-------|----------|-----------|-----------------|
| LSTM  | 0.42     | 0.65      | 24h            |
| LSTM+ | 0.38     | 0.59      | 24h            |
| Ensemble| 0.35    | 0.54      | 24h            |

## 📖 API Documentation

Detailed API documentation is available in [`docs/pvforecast_api_doc.pdf`](docs/pvforecast_api_doc.pdf)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. 🍴 Fork the repository
2. 🌿 Create your feature branch (`git checkout -b feature/amazing-feature`)
3. ✍ Commit your changes (`git commit -m 'Add amazing feature'`)
4. 📤 Push to the branch (`git push origin feature/amazing-feature`)
5. 🔄 Open a Merge Request

## 📋 Project Status

- ✅ Core ML models implemented
- ✅ Data pipeline operational
- 🚧 Web interface under development
- 📝 API documentation in progress
- 🔄 Model optimization ongoing

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ☁️ Weather Data Source

The weather forecast data is provided by GeoSphere Austria's AROME (Application of Research to Operations at MEsoscale) model with the following specifications:

- **Spatial Resolution**: 2.5 km grid
- **Temporal Resolution**: Hourly data
- **Update Frequency**: Recalculated every 3 hours
- **Forecast Horizon**: 60 hours
- **Geographic Coverage**: 42.98° - 51.82° N, 5.49° - 22.1° E (extended Alpine region)
- **Projection**: WGS84 - World Geodetic System 1984 (EPSG: 4326)
- **Parameters**: Temperature, precipitation, wind, global radiation, relative humidity, thunderstorm indices, cloud cover, pressure
- **License**: Creative Commons Attribution 4.0
- **DOI**: https://doi.org/10.60669/9zm8-s664
- **Model Development**: The AROME model code is developed in collaboration with partner weather services of the ACCORD consortium

## 📧 Contact

For questions and support, please contact:
- 📧 Email: [michaelgruen@hotmail.com]
