# 📈 Stock Market Predictor

A comprehensive stock market prediction application built with Streamlit and LSTM neural networks.

## Features

- **Real-time Data Fetching**: Get live stock data from Yahoo Finance
- **Interactive UI**: User-friendly Streamlit interface
- **LSTM Neural Network**: Advanced time series prediction using PyTorch
- **Customizable Parameters**: Adjust prediction timeframes and model settings
- **Visual Analytics**: Interactive charts for historical and predicted data
- **Multiple Stocks**: Support for any stock symbol available on Yahoo Finance

## Installation

1. Clone this repository:
```bash
git clone <your-repo-url>
cd stock-market-predictor
```

2. Create a virtual environment (recommended):
```bash
python -m venv .venv
.venv\Scripts\activate  # On Windows
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run main.py
```

2. Open your browser and navigate to `http://localhost:8501`

3. Configure your prediction:
   - Enter a stock symbol (e.g., AAPL, GOOGL, TSLA)
   - Select time period for historical data
   - Adjust prediction parameters
   - Click "🚀 Run Prediction"

## How It Works

1. **Data Collection**: Fetches historical stock price data using Yahoo Finance API
2. **Data Preprocessing**: Normalizes prices and creates sequences for time series prediction
3. **Model Training**: Trains an LSTM neural network on historical patterns
4. **Prediction**: Uses the trained model to predict future stock prices

## Model Architecture

The application uses a 2-layer LSTM neural network with the following structure:
- Input: Sequence of normalized stock prices
- Hidden layers: 2 LSTM layers with 50 hidden units each
- Output: Single predicted price value

## Supported Stock Symbols

Any stock symbol available on Yahoo Finance, including:
- **Tech**: AAPL, GOOGL, MSFT, AMZN, META, TSLA
- **Finance**: JPM, BAC, WFC, GS
- **Healthcare**: JNJ, PFE, UNH, ABBV
- **And many more...**

## Customization Options

- **Time Period**: 6 months to 5 years of historical data
- **Prediction Days**: 1 to 30 days ahead
- **Sequence Length**: 30 to 120 days for pattern recognition
- **Training Epochs**: 10 to 100 epochs for model training

## Important Disclaimer

⚠️ **This application is for educational purposes only.** Stock market predictions are inherently uncertain and should not be used as the sole basis for investment decisions. Always consult with financial advisors and do your own research before making investment decisions.

## Technical Requirements

- Python 3.8+
- Internet connection for data fetching
- Minimum 4GB RAM recommended
- Modern web browser

## Troubleshooting

### Common Issues:

1. **"Symbol not found"**: Ensure you're using a valid Yahoo Finance symbol
2. **"Not enough data"**: Try a longer time period or different stock
3. **Slow training**: Reduce the number of epochs or sequence length
4. **Memory issues**: Close other applications or reduce data size

### Performance Tips:

- Use shorter time periods for faster processing
- Reduce epochs for quicker training (may affect accuracy)
- Close unnecessary browser tabs while running

## Future Enhancements

- [ ] Multiple stock comparison
- [ ] Technical indicators integration
- [ ] Model performance metrics
- [ ] Export predictions to CSV
- [ ] Advanced visualization options
- [ ] Real-time prediction updates

## Contributing

Feel free to contribute to this project by:
- Reporting bugs
- Suggesting features
- Submitting pull requests
- Improving documentation

## License

This project is licensed under the MIT License - see the LICENSE file for details.