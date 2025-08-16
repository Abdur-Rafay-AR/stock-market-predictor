import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
from typing import Tuple, Optional, List
import os
import pickle

warnings.filterwarnings('ignore')

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Simple LSTM model for price prediction
class StockLSTM(nn.Module):
    """LSTM for stock price prediction"""
    def __init__(self, input_size: int = 1, hidden_size: int = 50, num_layers: int = 2, dropout: float = 0.2):
        super(StockLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Apply dropout and get the last time step
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out


# Data Processing Functions
@st.cache_data
def fetch_stock_data(symbol: str, period: str = "2y") -> Optional[pd.DataFrame]:
    """
    Fetch stock data from Yahoo Finance
    
    Args:
        symbol (str): Stock symbol (e.g., 'AAPL')
        period (str): Time period for data (e.g., '1y', '2y')
    
    Returns:
        pd.DataFrame or None: Stock data with OHLCV columns
    """
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period=period)
        
        if data.empty:
            st.error(f"No data found for symbol: {symbol}")
            return None
            
        # Validate required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in data.columns for col in required_columns):
            st.error(f"Invalid data format for {symbol}")
            return None
            
        return data
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return None

@st.cache_data
def validate_stock_symbol(symbol: str) -> bool:
    """
    Validate if a stock symbol exists
    
    Args:
        symbol (str): Stock symbol to validate
    
    Returns:
        bool: True if symbol is valid, False otherwise
    """
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        return 'symbol' in info or 'shortName' in info
    except:
        return False

def prepare_data(data: pd.DataFrame, sequence_length: int = 60) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[MinMaxScaler]]:
    """
    Prepare data for LSTM model
    
    Args:
        data (pd.DataFrame): Stock data
        sequence_length (int): Length of input sequences
    
    Returns:
        Tuple containing X (features), y (targets), and scaler
    """
    if data is None or len(data) < sequence_length:
        return None, None, None
    
    try:
        # Use closing prices
        prices = data['Close'].values.reshape(-1, 1)
        
        # Check for any NaN or infinite values
        if np.any(np.isnan(prices)) or np.any(np.isinf(prices)):
            st.error("Data contains invalid values (NaN or infinity)")
            return None, None, None
        
        # Scale the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_prices = scaler.fit_transform(prices)
        
        # Create sequences
        X, y = [], []
        for i in range(sequence_length, len(scaled_prices)):
            X.append(scaled_prices[i-sequence_length:i])
            y.append(scaled_prices[i])
        
        if len(X) == 0:
            st.error("Not enough data to create training sequences")
            return None, None, None
        
        return np.array(X), np.array(y), scaler
    
    except Exception as e:
        st.error(f"Error preparing data: {str(e)}")
        return None, None, None

def create_simple_model(input_size: int = 1, hidden_size: int = 50, num_layers: int = 2) -> StockLSTM:
    """
    Create a simple LSTM model for demonstration
    
    Args:
        input_size (int): Number of input features
        hidden_size (int): Number of hidden units
        num_layers (int): Number of LSTM layers
    
    Returns:
        StockLSTM: Initialized model
    """
    model = StockLSTM(input_size, hidden_size, num_layers)
    return model

def train_simple_model(X: np.ndarray, y: np.ndarray, epochs: int = 50, learning_rate: float = 0.001) -> StockLSTM:
    """
    Train a simple LSTM model
    
    Args:
        X (np.ndarray): Training features
        y (np.ndarray): Training targets
        epochs (int): Number of training epochs
        learning_rate (float): Learning rate for optimizer
    
    Returns:
        StockLSTM: Trained model
    """
    model = create_simple_model()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)
    
    # Split data for validation
    split_idx = int(len(X_tensor) * 0.8)
    X_train, X_val = X_tensor[:split_idx], X_tensor[split_idx:]
    y_train, y_val = y_tensor[:split_idx], y_tensor[split_idx:]
    
    model.train()
    progress_bar = st.progress(0)
    status_text = st.empty()
    loss_history = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        train_loss = criterion(outputs, y_train)
        train_loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)
        
        loss_history.append((train_loss.item(), val_loss.item()))
        
        # Update progress
        progress = (epoch + 1) / epochs
        progress_bar.progress(progress)
        status_text.text(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss.item():.6f} - Val Loss: {val_loss.item():.6f}')
    
    progress_bar.empty()
    status_text.empty()
    
    # Display training results
    if len(loss_history) > 0:
        train_losses, val_losses = zip(*loss_history)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(epochs)), y=train_losses, name='Training Loss'))
        fig.add_trace(go.Scatter(x=list(range(epochs)), y=val_losses, name='Validation Loss'))
        fig.update_layout(title="Training Progress", xaxis_title="Epoch", yaxis_title="Loss")
        st.plotly_chart(fig, use_container_width=True)
    
    return model

def predict_future_prices(model: StockLSTM, last_sequence: np.ndarray, scaler: MinMaxScaler, days: int = 7) -> np.ndarray:
    """
    Predict future stock prices
    
    Args:
        model (StockLSTM): Trained model
        last_sequence (np.ndarray): Last sequence from training data
        scaler (MinMaxScaler): Fitted scaler for inverse transformation
        days (int): Number of days to predict
    
    Returns:
        np.ndarray: Predicted prices
    """
    model.eval()
    predictions = []
    current_sequence = last_sequence.copy()
    
    with torch.no_grad():
        for _ in range(days):
            # Predict next price
            input_tensor = torch.FloatTensor(current_sequence).unsqueeze(0)
            prediction = model(input_tensor)
            predictions.append(prediction.item())
            
            # Update sequence for next prediction
            new_sequence = np.append(current_sequence[1:], [[prediction.item()]], axis=0)
            current_sequence = new_sequence
    
    # Inverse transform predictions
    predictions = np.array(predictions).reshape(-1, 1)
    predictions = scaler.inverse_transform(predictions)
    
    return predictions.flatten()

def save_model(model: StockLSTM, scaler: MinMaxScaler, symbol: str) -> str:
    """
    Save trained model and scaler to disk
    
    Args:
        model (StockLSTM): Trained model
        scaler (MinMaxScaler): Fitted scaler
        symbol (str): Stock symbol
    
    Returns:
        str: Path to saved model
    """
    try:
        model_dir = "saved_models"
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, f"{symbol}_model.pth")
        scaler_path = os.path.join(model_dir, f"{symbol}_scaler.pkl")
        
        # Save model state dict
        torch.save(model.state_dict(), model_path)
        
        # Save scaler
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        return model_path
    except Exception as e:
        st.error(f"Error saving model: {str(e)}")
        return ""

def load_model(symbol: str) -> Tuple[Optional[StockLSTM], Optional[MinMaxScaler]]:
    """
    Load saved model and scaler from disk
    
    Args:
        symbol (str): Stock symbol
    
    Returns:
        Tuple containing loaded model and scaler, or (None, None) if not found
    """
    try:
        model_dir = "saved_models"
        model_path = os.path.join(model_dir, f"{symbol}_model.pth")
        scaler_path = os.path.join(model_dir, f"{symbol}_scaler.pkl")
        
        if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
            return None, None
        
        # Load model
        model = create_simple_model()
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        # Load scaler
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None
# -------------------------------
# Streamlit UI
# -------------------------------
def main():
    st.set_page_config(
        page_title="Stock Market Predictor",
        page_icon="📈",
        layout="wide"
    )
    
    st.title("📈 Stock Market Predictor")
    st.markdown("*Predict stock prices using LSTM Neural Networks*")
    
    # Sidebar for inputs
    st.sidebar.header("Configuration")
    
    # Stock symbol input
    symbol = st.sidebar.text_input(
        "Enter Stock Symbol", 
        value="AAPL",
        help="Enter a valid stock symbol (e.g., AAPL, GOOGL, TSLA)"
    ).upper().strip()
    
    # Validate symbol format
    if symbol and not symbol.replace('.', '').replace('-', '').isalnum():
        st.sidebar.error("Invalid symbol format")
        symbol = ""
    
    # Time period selection
    period_options = {
        "6 months": "6mo",
        "1 year": "1y", 
        "2 years": "2y",
        "5 years": "5y"
    }
    period_display = st.sidebar.selectbox("Select Time Period", list(period_options.keys()))
    period = period_options[period_display]
    
    # Prediction settings
    prediction_days = st.sidebar.slider("Days to Predict", 1, 30, 7)
    sequence_length = st.sidebar.slider("Sequence Length", 30, 120, 60)
    
    # Training settings
    st.sidebar.subheader("Model Training")
    train_epochs = st.sidebar.slider("Training Epochs", 10, 100, 50)
    learning_rate = st.sidebar.selectbox("Learning Rate", [0.01, 0.001, 0.0001], index=1)
    
    # Model persistence options
    use_saved_model = st.sidebar.checkbox("Use saved model (if available)", value=True)
    save_new_model = st.sidebar.checkbox("Save trained model", value=True)
    
    # Check if saved model exists
    saved_model_exists = False
    if symbol and use_saved_model:
        model_dir = "saved_models"
        if os.path.exists(os.path.join(model_dir, f"{symbol}_model.pth")):
            saved_model_exists = True
            st.sidebar.success(f"✅ Saved model found for {symbol}")
    
    if st.sidebar.button("🚀 Run Prediction", type="primary") and symbol:
        with st.spinner("Fetching stock data..."):
            # Fetch data
            data = fetch_stock_data(symbol, period)
            
            if data is not None and len(data) > sequence_length:
                st.success(f"Successfully fetched {len(data)} days of data for {symbol}")
                
                # Display stock info
                col1, col2, col3, col4 = st.columns(4)
                latest_price = data['Close'].iloc[-1]
                price_change = data['Close'].iloc[-1] - data['Close'].iloc[-2]
                price_change_pct = (price_change / data['Close'].iloc[-2]) * 100
                volume = data['Volume'].iloc[-1]
                
                with col1:
                    st.metric("Latest Price", f"${latest_price:.2f}")
                with col2:
                    st.metric("Price Change", f"${price_change:.2f}", f"{price_change_pct:.2f}%")
                with col3:
                    st.metric("Volume", f"{volume:,.0f}")
                with col4:
                    st.metric("Data Points", len(data))
                
                # Plot historical data
                st.subheader("📊 Historical Stock Price")
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data['Close'],
                    mode='lines',
                    name=f'{symbol} Close Price',
                    line=dict(color='blue', width=2)
                ))
                fig.update_layout(
                    title=f"{symbol} Stock Price History",
                    xaxis_title="Date",
                    yaxis_title="Price ($)",
                    hovermode='x'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Prepare data for training
                st.subheader("🤖 Model Training & Prediction")
                
                # Try to load saved model first
                model, scaler = None, None
                if use_saved_model and saved_model_exists:
                    with st.spinner("Loading saved model..."):
                        model, scaler = load_model(symbol)
                        if model is not None:
                            st.success(f"✅ Loaded saved model for {symbol}")
                
                # If no saved model or loading failed, train new model
                if model is None or scaler is None:
                    with st.spinner("Preparing data and training model..."):
                        X, y, scaler = prepare_data(data, sequence_length)
                        
                        if X is not None:
                            # Train model
                            st.info("Training LSTM model...")
                            model = train_simple_model(X, y, train_epochs, learning_rate)
                            
                            # Save model if requested
                            if save_new_model:
                                save_path = save_model(model, scaler, symbol)
                                if save_path:
                                    st.success(f"✅ Model saved to {save_path}")
                        else:
                            st.error("Failed to prepare data for training")
                            return
                else:
                    # Still need to prepare data for predictions
                    X, y, _ = prepare_data(data, sequence_length)
                    if X is None:
                        st.error("Failed to prepare data for predictions")
                        return
                        
                # Make predictions
                if model is not None and scaler is not None and X is not None:
                    st.info("Generating predictions...")
                    last_sequence = X[-1]
                    future_predictions = predict_future_prices(
                        model, last_sequence, scaler, prediction_days
                    )
                        
                    # Create prediction dates
                    last_date = data.index[-1]
                    future_dates = [last_date + timedelta(days=i+1) for i in range(prediction_days)]
                    
                    # Display predictions
                    st.subheader("🔮 Future Price Predictions")
                    
                    # Create prediction dataframe
                    pred_df = pd.DataFrame({
                        'Date': future_dates,
                        'Predicted Price': future_predictions,
                        'Day': [f"Day {i+1}" for i in range(prediction_days)]
                    })
                    
                    # Display prediction table
                    st.dataframe(pred_df, use_container_width=True)
                    
                    # Plot predictions
                    fig_pred = go.Figure()
                    
                    # Historical data (last 30 days)
                    recent_data = data.tail(30)
                    fig_pred.add_trace(go.Scatter(
                        x=recent_data.index,
                        y=recent_data['Close'],
                        mode='lines',
                        name='Historical Prices',
                        line=dict(color='blue', width=2)
                    ))
                    
                    # Predictions
                    fig_pred.add_trace(go.Scatter(
                        x=future_dates,
                        y=future_predictions,
                        mode='lines+markers',
                        name='Predictions',
                        line=dict(color='red', width=2, dash='dash'),
                        marker=dict(size=8)
                    ))
                    
                    # Connect last historical point to first prediction
                    fig_pred.add_trace(go.Scatter(
                        x=[recent_data.index[-1], future_dates[0]],
                        y=[recent_data['Close'].iloc[-1], future_predictions[0]],
                        mode='lines',
                        name='Connection',
                        line=dict(color='orange', width=2, dash='dot'),
                        showlegend=False
                    ))
                    
                    fig_pred.update_layout(
                        title=f"{symbol} Price Prediction",
                        xaxis_title="Date",
                        yaxis_title="Price ($)",
                        hovermode='x'
                    )
                    st.plotly_chart(fig_pred, use_container_width=True)
                    
                    # Prediction summary
                    current_price = data['Close'].iloc[-1]
                    predicted_price = future_predictions[-1]
                    price_difference = predicted_price - current_price
                    percentage_change = (price_difference / current_price) * 100
                    
                    st.subheader("📋 Prediction Summary")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            f"Current Price", 
                            f"${current_price:.2f}"
                        )
                    with col2:
                        st.metric(
                            f"Predicted Price ({prediction_days} days)", 
                            f"${predicted_price:.2f}",
                            f"${price_difference:.2f}"
                        )
                    with col3:
                        st.metric(
                            "Expected Change", 
                            f"{percentage_change:.2f}%",
                            f"${price_difference:.2f}"
                        )
                    
                    # Risk disclaimer
                    st.warning("""
                    ⚠️ **Investment Disclaimer**: This prediction is for educational purposes only. 
                    Stock market predictions are inherently uncertain and should not be used as the sole basis for investment decisions. 
                    Always consult with financial advisors and do your own research before making investment decisions.
                    """)
                else:
                    st.error("Model training failed or insufficient data.")
            else:
                st.error(f"Could not fetch sufficient data for {symbol}. Please check the symbol and try again.")
    elif not symbol:
        st.sidebar.warning("Please enter a stock symbol to begin.")
    
    # Information section
    with st.expander("ℹ️ How it works"):
        st.markdown("""
        ### Stock Market Prediction using LSTM Neural Networks
        
        **How the model works:**
        1. **Data Collection**: Fetches historical stock price data using Yahoo Finance API
        2. **Data Preprocessing**: Normalizes prices and creates sequences for time series prediction
        3. **Model Training**: Trains an LSTM (Long Short-Term Memory) neural network on historical patterns
        4. **Prediction**: Uses the trained model to predict future stock prices
        
        **Key Features:**
        - Real-time data fetching from Yahoo Finance
        - Interactive parameter tuning
        - Visual charts for both historical and predicted data
        - Customizable prediction timeframes
        
        **Limitations:**
        - Predictions are based on historical patterns and may not account for market volatility
        - Model performance depends on data quality and market conditions
        - Should be used for educational purposes only
        """)

if __name__ == "__main__":
    main()
