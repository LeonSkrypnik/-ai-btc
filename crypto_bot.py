"""
COMPLETE CRYPTO TRADING BOT WITH GITHUB ACTIONS INTEGRATION
Author: AI Assistant
Description: Full automated trading bot for meme coins with model training
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import requests
from dataclasses import dataclass, asdict

# Machine Learning Imports
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb

# Deep Learning Imports
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model, Input
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not available, using XGBoost only")

# Database (optional) - SQLite for simplicity
import sqlite3
from contextlib import closing

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION SECTION
# ============================================================================
@dataclass
class Config:
    """Configuration class for the trading bot"""
    # API Configuration
    COINAPI_KEY: str = os.getenv('COINAPI_KEY', 'YOUR_API_KEY_HERE')
    COINGECKO_API_KEY: str = os.getenv('COINGECKO_API_KEY', '')
    
    # Trading Parameters
    MEME_COINS: List[str] = None  # Will be populated dynamically
    BASE_CURRENCY: str = 'USDT'
    TIMEFRAME: str = '4HRS'  # 1HRS, 4HRS, 1DAY
    LOOKBACK_PERIODS: int = 500
    
    # Model Parameters
    SEQUENCE_LENGTH: int = 60
    TRAIN_TEST_SPLIT: float = 0.8
    EPOCHS: int = 50
    BATCH_SIZE: int = 32
    
    # Risk Management
    MAX_POSITION_SIZE: float = 0.1  # 10% of portfolio per trade
    STOP_LOSS_PCT: float = 0.05  # 5% stop loss
    TAKE_PROFIT_PCT: float = 0.15  # 15% take profit
    
    # GitHub Actions
    GITHUB_ACTIONS: bool = os.getenv('GITHUB_ACTIONS', 'false').lower() == 'true'
    OUTPUT_DIR: str = 'results' if not GITHUB_ACTIONS else '/github/workspace/results'
    
    def __post_init__(self):
        """Initialize default meme coins list"""
        if self.MEME_COINS is None:
            self.MEME_COINS = ['DOGE', 'SHIB', 'PEPE', 'FLOKI', 'BONK', 'WIF', 'MEME', 'BABYDOGE']

# ============================================================================
# DATA COLLECTION MODULE
# ============================================================================
class CryptoDataCollector:
    """Handles all data collection from various APIs"""
    
    def __init__(self, config: Config):
        self.config = config
        self.coinapi_headers = {'X-CoinAPI-Key': config.COINAPI_KEY}
        self.coinapi_base = 'https://rest.coinapi.io/v1'
        
        # Cache for API responses
        self._cache = {}
        
    def fetch_memecoins_dynamic(self) -> List[str]:
        """Fetch trending meme coins from CoinGecko"""
        try:
            url = "https://api.coingecko.com/api/v3/coins/markets"
            params = {
                'vs_currency': 'usd',
                'category': 'meme-token',
                'order': 'market_cap_desc',
                'per_page': 20,
                'page': 1
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                coins = response.json()
                symbols = [coin['symbol'].upper() for coin in coins]
                print(f"Fetched {len(symbols)} meme coins from CoinGecko")
                return symbols[:10]  # Top 10
        except Exception as e:
            print(f"Error fetching from CoinGecko: {e}")
        
        # Fallback to default list
        return self.config.MEME_COINS
    
    def get_coinapi_symbol_id(self, coin: str) -> Optional[str]:
        """Convert coin symbol to CoinAPI symbol ID"""
        # Try different exchange formats
        exchanges = ['BINANCE', 'COINBASE', 'KRAKEN', 'HUOBI']
        for exchange in exchanges:
            symbol_id = f"{exchange}_SPOT_{coin}_{self.config.BASE_CURRENCY}"
            
            # Quick check if symbol exists
            url = f"{self.coinapi_base}/ohlcv/{symbol_id}/latest"
            response = requests.get(url, headers=self.coinapi_headers, params={'period_id': '1HRS'})
            
            if response.status_code == 200:
                return symbol_id
        
        return None
    
    def fetch_ohlcv_data(self, coin: str) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data for a specific coin"""
        
        cache_key = f"{coin}_{self.config.TIMEFRAME}"
        if cache_key in self._cache:
            return self._cache[cache_key].copy()
        
        symbol_id = self.get_coinapi_symbol_id(coin)
        if not symbol_id:
            print(f"No valid symbol ID found for {coin}")
            return None
        
        url = f"{self.coinapi_base}/ohlcv/{symbol_id}/history"
        
        # Calculate time range
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=self.config.LOOKBACK_PERIODS * 
                                         {'1HRS': 1/24, '4HRS': 4/24, '1DAY': 1}[self.config.TIMEFRAME])
        
        params = {
            'period_id': self.config.TIMEFRAME,
            'time_start': start_time.isoformat() + 'Z',
            'time_end': end_time.isoformat() + 'Z',
            'limit': self.config.LOOKBACK_PERIODS
        }
        
        try:
            print(f"Fetching data for {coin}...")
            response = requests.get(url, headers=self.coinapi_headers, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            if not data:
                print(f"No data returned for {coin}")
                return None
            
            df = pd.DataFrame(data)
            
            # Process and clean data
            if 'time_period_end' in df.columns:
                df['time_period_end'] = pd.to_datetime(df['time_period_end'])
                df.set_index('time_period_end', inplace=True)
            
            # Rename columns to standard format
            column_map = {
                'price_open': 'open',
                'price_high': 'high',
                'price_low': 'low',
                'price_close': 'close',
                'volume_traded': 'volume'
            }
            df.rename(columns=column_map, inplace=True)
            
            # Keep only necessary columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            df = df[[col for col in required_cols if col in df.columns]]
            
            # Remove any rows with NaN
            df.dropna(inplace=True)
            
            if len(df) < 100:
                print(f"Insufficient data for {coin}: {len(df)} rows")
                return None
            
            # Cache the result
            self._cache[cache_key] = df.copy()
            
            print(f"Successfully fetched {len(df)} rows for {coin}")
            return df
            
        except requests.exceptions.RequestException as e:
            print(f"API Error for {coin}: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error for {coin}: {e}")
            return None
    
    def fetch_all_coins_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch data for all meme coins"""
        print("\n" + "="*50)
        print("STARTING DATA COLLECTION")
        print("="*50)
        
        # Get dynamic list of meme coins
        coins = self.fetch_memecoins_dynamic()
        print(f"Processing {len(coins)} coins: {coins}")
        
        all_data = {}
        
        for coin in coins:
            df = self.fetch_ohlcv_data(coin)
            if df is not None and len(df) > 100:
                all_data[coin] = df
                print(f"✓ {coin}: {len(df)} rows")
            else:
                print(f"✗ {coin}: Failed or insufficient data")
        
        print(f"\nSuccessfully collected data for {len(all_data)} coins")
        return all_data

# ============================================================================
# FEATURE ENGINEERING MODULE
# ============================================================================
class FeatureEngineer:
    """Creates technical indicators and features for prediction"""
    
    @staticmethod
    def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive technical indicators"""
        df = df.copy()
        
        # Price-based features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Moving averages
        df['sma_5'] = df['close'].rolling(5).mean()
        df['sma_10'] = df['close'].rolling(10).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        
        # Exponential moving averages
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # ATR (Average True Range)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['atr'] = true_range.rolling(14).mean()
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        
        # Price patterns
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']
        df['body_size'] = np.abs(df['close'] - df['open']) / (df['high'] - df['low']).replace(0, 1)
        
        # Volatility
        df['volatility'] = df['returns'].rolling(20).std()
        df['rolling_max'] = df['close'].rolling(20).max()
        df['rolling_min'] = df['close'].rolling(20).min()
        df['distance_to_high'] = (df['rolling_max'] - df['close']) / df['rolling_max']
        df['distance_to_low'] = (df['close'] - df['rolling_min']) / df['close']
        
        # Statistical features
        df['z_score'] = (df['close'] - df['close'].rolling(20).mean()) / df['close'].rolling(20).std()
        df['skew'] = df['returns'].rolling(20).skew()
        df['kurtosis'] = df['returns'].rolling(20).kurt()
        
        # Target: Future return (next period)
        df['target'] = df['close'].shift(-1) / df['close'] - 1
        
        # Drop NaN values
        df.dropna(inplace=True)
        
        return df
    
    @staticmethod
    def create_sequences(data: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series prediction"""
        sequences = []
        targets = []
        
        for i in range(len(data) - sequence_length):
            seq = data[i:i + sequence_length]
            target = data[i + sequence_length, -1]  # Last column is target
            sequences.append(seq)
            targets.append(target)
        
        return np.array(sequences), np.array(targets)

# ============================================================================
# MACHINE LEARNING MODELS
# ============================================================================
class LSTMModel:
    """LSTM-based prediction model"""
    
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.model = self._build_model()
        
    def _build_model(self):
        """Build LSTM model with attention"""
        inputs = Input(shape=self.input_shape)
        
        # First LSTM layer
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(inputs)
        x = layers.Dropout(0.3)(x)
        
        # Second LSTM layer
        x = layers.Bidirectional(layers.LSTM(32, return_sequences=True))(x)
        x = layers.Dropout(0.2)(x)
        
        # Attention mechanism
        attention = layers.Dense(1, activation='tanh')(x)
        attention = layers.Flatten()(attention)
        attention = layers.Activation('softmax')(attention)
        attention = layers.RepeatVector(32 * 2)(attention)  # 2 for bidirectional
        attention = layers.Permute([2, 1])(attention)
        
        # Apply attention
        x = layers.Multiply()([x, attention])
        x = layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(x)
        
        # Dense layers
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(16, activation='relu')(x)
        
        # Output layer
        outputs = layers.Dense(1)(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='huber',
            metrics=['mae', 'mse']
        )
        
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """Train the model"""
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0
        )
        
        return history
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X, verbose=0).flatten()

class XGBoostModel:
    """XGBoost-based prediction model"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        
    def train(self, X_train, y_train, X_val, y_val):
        """Train XGBoost model"""
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Prepare DMatrix
        dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
        dval = xgb.DMatrix(X_val_scaled, label=y_val)
        
        # Parameters
        params = {
            'objective': 'reg:squarederror',
            'learning_rate': 0.01,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'gamma': 0,
            'alpha': 0.1,
            'lambda': 1,
            'eval_metric': ['mae', 'rmse']
        }
        
        # Train with early stopping
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=50,
            verbose_eval=False
        )
        
        # Make predictions
        train_pred = self.predict(X_train)
        val_pred = self.predict(X_val)
        
        return train_pred, val_pred
    
    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        X_scaled = self.scaler.transform(X)
        dtest = xgb.DMatrix(X_scaled)
        return self.model.predict(dtest)

# ============================================================================
# ENSEMBLE MODEL
# ============================================================================
class EnsembleTradingModel:
    """Combines multiple models for better predictions"""
    
    def __init__(self, config: Config):
        self.config = config
        self.lstm_model = None
        self.xgb_model = None
        self.scalers = {}
        self.feature_columns = None
        
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, list]:
        """Prepare data for model training"""
        # Engineer features
        df_with_features = FeatureEngineer.add_technical_indicators(df)
        
        # Separate features and target
        feature_cols = [col for col in df_with_features.columns if col != 'target']
        self.feature_columns = feature_cols
        
        X = df_with_features[feature_cols].values
        y = df_with_features['target'].values
        
        # Scale features
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers['features'] = scaler
        
        # Scale target
        target_scaler = RobustScaler()
        y_scaled = target_scaler.fit_transform(y.reshape(-1, 1)).flatten()
        self.scalers['target'] = target_scaler
        
        return X_scaled, y_scaled, feature_cols
    
    def train(self, df: pd.DataFrame) -> Dict:
        """Train ensemble model"""
        print(f"\nTraining model on {len(df)} samples...")
        
        # Prepare data
        X_scaled, y_scaled, feature_cols = self.prepare_data(df)
        
        # Split data
        split_idx = int(len(X_scaled) * self.config.TRAIN_TEST_SPLIT)
        X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_test = y_scaled[:split_idx], y_scaled[split_idx:]
        
        # Create sequences for LSTM
        X_train_seq, y_train_seq = FeatureEngineer.create_sequences(
            np.column_stack([X_train, y_train]), 
            self.config.SEQUENCE_LENGTH
        )
        X_test_seq, y_test_seq = FeatureEngineer.create_sequences(
            np.column_stack([X_test, y_test]), 
            self.config.SEQUENCE_LENGTH
        )
        
        results = {}
        
        # Train XGBoost (on non-sequential data)
        if len(X_train) > 100:
            print("Training XGBoost model...")
            self.xgb_model = XGBoostModel()
            train_pred_xgb, test_pred_xgb = self.xgb_model.train(
                X_train, y_train, X_test, y_test
            )
            
            # Store results
            results['xgb'] = {
                'train_pred': train_pred_xgb,
                'test_pred': test_pred_xgb,
                'train_mae': mean_absolute_error(y_train, train_pred_xgb),
                'test_mae': mean_absolute_error(y_test, test_pred_xgb)
            }
            print(f"XGBoost Test MAE: {results['xgb']['test_mae']:.6f}")
        
        # Train LSTM
        if TF_AVAILABLE and len(X_train_seq) > 0:
            print("Training LSTM model...")
            self.lstm_model = LSTMModel((X_train_seq.shape[1], X_train_seq.shape[2]))
            
            # Split sequences for validation
            val_split = 0.2
            val_size = int(len(X_train_seq) * val_split)
            X_train_lstm, X_val_lstm = X_train_seq[:-val_size], X_train_seq[-val_size:]
            y_train_lstm, y_val_lstm = y_train_seq[:-val_size], y_train_seq[-val_size:]
            
            history = self.lstm_model.train(
                X_train_lstm, y_train_lstm,
                X_val_lstm, y_val_lstm,
                epochs=self.config.EPOCHS,
                batch_size=self.config.BATCH_SIZE
            )
            
            # Make predictions
            train_pred_lstm = self.lstm_model.predict(X_train_lstm)
            test_pred_lstm = self.lstm_model.predict(X_test_seq)
            
            results['lstm'] = {
                'train_pred': train_pred_lstm,
                'test_pred': test_pred_lstm,
                'train_mae': mean_absolute_error(y_train_lstm, train_pred_lstm),
                'test_mae': mean_absolute_error(y_test_seq, test_pred_lstm),
                'history': history.history
            }
            print(f"LSTM Test MAE: {results['lstm']['test_mae']:.6f}")
        
        # Ensemble predictions
        if 'xgb' in results and 'lstm' in results:
            # Align predictions (LSTM has fewer samples due to sequences)
            lstm_start = len(y_test) - len(results['lstm']['test_pred'])
            xgb_pred_aligned = results['xgb']['test_pred'][lstm_start:]
            lstm_pred = results['lstm']['test_pred']
            y_test_aligned = y_test[lstm_start:]
            
            # Weighted ensemble
            weights = {'xgb': 0.6, 'lstm': 0.4}
            ensemble_pred = (weights['xgb'] * xgb_pred_aligned + 
                           weights['lstm'] * lstm_pred)
            
            results['ensemble'] = {
                'test_pred': ensemble_pred,
                'test_mae': mean_absolute_error(y_test_aligned, ensemble_pred),
                'weights': weights
            }
            print(f"Ensemble Test MAE: {results['ensemble']['test_mae']:.6f}")
        
        return results
    
    def predict_next(self, df: pd.DataFrame) -> Dict:
        """Predict next period's return"""
        # Prepare latest data
        df_with_features = FeatureEngineer.add_technical_indicators(df)
        
        if self.feature_columns is None:
            self.feature_columns = [col for col in df_with_features.columns if col != 'target']
        
        # Get latest sequence
        X_latest = df_with_features[self.feature_columns].iloc[-self.config.SEQUENCE_LENGTH:].values
        
        # Scale
        if 'features' in self.scalers:
            X_scaled = self.scalers['features'].transform(X_latest)
        else:
            X_scaled = X_latest
        
        predictions = {}
        
        # XGBoost prediction (using last row only)
        if self.xgb_model is not None:
            xgb_pred = self.xgb_model.predict(X_latest[-1:].reshape(1, -1))[0]
            predictions['xgb'] = float(xgb_pred)
        
        # LSTM prediction
        if self.lstm_model is not None:
            X_seq = X_scaled.reshape(1, self.config.SEQUENCE_LENGTH, -1)
            lstm_pred = self.lstm_model.predict(X_seq)[0]
            predictions['lstm'] = float(lstm_pred)
        
        # Ensemble prediction
        if 'xgb' in predictions and 'lstm' in predictions:
            weights = {'xgb': 0.6, 'lstm': 0.4}
            ensemble_pred = (weights['xgb'] * predictions['xgb'] + 
                           weights['lstm'] * predictions['lstm'])
            predictions['ensemble'] = float(ensemble_pred)
            
            # Convert back from scaled value
            if 'target' in self.scalers:
                # Create fake array for inverse transform
                pred_array = np.array([[ensemble_pred]])
                ensemble_pred_original = self.scalers['target'].inverse_transform(pred_array)[0][0]
                predictions['ensemble_original'] = float(ensemble_pred_original)
        
        # Generate trading signal
        if 'ensemble_original' in predictions:
            signal = self._generate_signal(predictions['ensemble_original'])
            predictions['signal'] = signal
            predictions['confidence'] = abs(predictions['ensemble_original'])
        
        return predictions
    
    def _generate_signal(self, predicted_return: float) -> str:
        """Generate trading signal based on predicted return"""
        if predicted_return > 0.02:  # > 2% predicted gain
            return 'STRONG_BUY'
        elif predicted_return > 0.005:  # > 0.5% predicted gain
            return 'BUY'
        elif predicted_return < -0.02:  # < -2% predicted loss
            return 'STRONG_SELL'
        elif predicted_return < -0.005:  # < -0.5% predicted loss
            return 'SELL'
        else:
            return 'HOLD'

# ============================================================================
# TRADING BOT CORE
# ============================================================================
class CryptoTradingBot:
    """Main trading bot class"""
    
    def __init__(self, config: Config):
        self.config = config
        self.data_collector = CryptoDataCollector(config)
        self.models = {}
        self.results = {}
        
        # Create output directory
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        
        # Initialize database
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database for storing results"""
        self.db_path = os.path.join(self.config.OUTPUT_DIR, 'trading_bot.db')
        
        with closing(sqlite3.connect(self.db_path)) as conn:
            cursor = conn.cursor()
            
            # Create tables
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    coin TEXT,
                    signal TEXT,
                    predicted_return REAL,
                    confidence REAL,
                    current_price REAL,
                    model_version TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    coin TEXT,
                    model_type TEXT,
                    mae REAL,
                    mse REAL,
                    train_size INTEGER,
                    test_size INTEGER
                )
            ''')
            
            conn.commit()
    
    def save_prediction(self, coin: str, prediction: Dict, current_price: float):
        """Save prediction to database"""
        with closing(sqlite3.connect(self.db_path)) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO predictions 
                (timestamp, coin, signal, predicted_return, confidence, current_price, model_version)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.utcnow().isoformat(),
                coin,
                prediction.get('signal', 'HOLD'),
                prediction.get('ensemble_original', 0),
                prediction.get('confidence', 0),
                current_price,
                'v1.0'
            ))
            
            conn.commit()
    
    def save_performance(self, coin: str, results: Dict):
        """Save model performance to database"""
        with closing(sqlite3.connect(self.db_path)) as conn:
            cursor = conn.cursor()
            
            for model_type, model_results in results.items():
                if 'test_mae' in model_results:
                    cursor.execute('''
                        INSERT INTO model_performance 
                        (timestamp, coin, model_type, mae, mse, train_size, test_size)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        datetime.utcnow().isoformat(),
                        coin,
                        model_type,
                        model_results.get('test_mae', 0),
                        model_results.get('test_mse', 0),
                        model_results.get('train_size', 0),
                        model_results.get('test_size', 0)
                    ))
            
            conn.commit()
    
    def run_for_coin(self, coin: str, df: pd.DataFrame) -> bool:
        """Run full pipeline for a single coin"""
        try:
            print(f"\n{'='*60}")
            print(f"PROCESSING: {coin}")
            print(f"{'='*60}")
            
            # Train model
            model = EnsembleTradingModel(self.config)
            results = model.train(df)
            
            # Make prediction for next period
            latest_prediction = model.predict_next(df)
            
            # Store model and results
            self.models[coin] = model
            self.results[coin] = {
                'performance': results,
                'prediction': latest_prediction,
                'latest_price': df['close'].iloc[-1],
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Save to database
            self.save_prediction(coin, latest_prediction, df['close'].iloc[-1])
            self.save_performance(coin, results)
            
            # Print results
            print(f"\nLatest Price: ${df['close'].iloc[-1]:.6f}")
            print(f"Predicted Return: {latest_prediction.get('ensemble_original', 0)*100:.2f}%")
            print(f"Trading Signal: {latest_prediction.get('signal', 'HOLD')}")
            
            if 'ensemble' in results:
                print(f"Model Performance (MAE): {results['ensemble']['test_mae']:.6f}")
            
            return True
            
        except Exception as e:
            print(f"Error processing {coin}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def generate_report(self):
        """Generate summary report"""
        report = {
            'timestamp': datetime.utcnow().isoformat(),
            'coins_processed': list(self.models.keys()),
            'successful_coins': len(self.models),
            'predictions': {},
            'summary': {}
        }
        
        buy_signals = []
        sell_signals = []
        
        for coin, data in self.results.items():
            report['predictions'][coin] = data['prediction']
            
            signal = data['prediction'].get('signal', 'HOLD')
            if 'BUY' in signal:
                buy_signals.append(coin)
            elif 'SELL' in signal:
                sell_signals.append(coin)
        
        report['summary'] = {
            'total_coins': len(self.results),
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'hold_signals': len(self.results) - len(buy_signals) - len(sell_signals),
            'strong_buy_count': len([c for c in buy_signals if 
                                   self.results[c]['prediction'].get('signal') == 'STRONG_BUY']),
            'avg_confidence': np.mean([d['prediction'].get('confidence', 0) 
                                      for d in self.results.values()])
        }
        
        # Save report
        report_path = os.path.join(self.config.OUTPUT_DIR, 'trading_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Print summary
        print(f"\n{'='*60}")
        print("TRADING REPORT SUMMARY")
        print(f"{'='*60}")
        print(f"Timestamp: {report['timestamp']}")
        print(f"Coins Processed: {report['summary']['total_coins']}")
        print(f"Buy Signals: {len(report['summary']['buy_signals'])} {buy_signals}")
        print(f"Sell Signals: {len(report['summary']['sell_signals'])} {sell_signals}")
        print(f"Strong Buy Recommendations: {report['summary']['strong_buy_count']}")
        print(f"Average Confidence: {report['summary']['avg_confidence']*100:.1f}%")
        
        return report
    
    def run(self):
        """Main execution method"""
        print("\n" + "="*60)
        print("CRYPTO TRADING BOT - FULL EXECUTION")
        print("="*60)
        
        start_time = datetime.now()
        
        try:
            # Step 1: Collect data
            all_data = self.data_collector.fetch_all_coins_data()
            
            if not all_data:
                print("No data collected. Exiting.")
                return
            
            # Step 2: Process each coin
            successful = 0
            for coin, df in all_data.items():
                if self.run_for_coin(coin, df):
                    successful += 1
            
            # Step 3: Generate report
            if successful > 0:
                self.generate_report()
                
                # Save models if needed (in practice, you might save to cloud)
                models_path = os.path.join(self.config.OUTPUT_DIR, 'models_summary.json')
                models_summary = {
                    coin: {
                        'features': list(self.models[coin].feature_columns) if 
                                   self.models[coin].feature_columns else None,
                        'last_trained': datetime.utcnow().isoformat()
                    }
                    for coin in self.models
                }
                
                with open(models_path, 'w') as f:
                    json.dump(models_summary, f, indent=2)
            
            # Step 4: Execution time
            elapsed = datetime.now() - start_time
            print(f"\nTotal execution time: {elapsed.total_seconds():.1f} seconds")
            print(f"Successfully processed {successful}/{len(all_data)} coins")
            
            # For GitHub Actions: Set output
            if self.config.GITHUB_ACTIONS:
                with open(os.environ['GITHUB_OUTPUT'], 'a') as f:
                    f.write(f"processed_coins={successful}\n")
                    f.write(f"elapsed_seconds={elapsed.total_seconds():.1f}\n")
            
        except Exception as e:
            print(f"\nFatal error in main execution: {e}")
            import traceback
            traceback.print_exc()
            
            if self.config.GITHUB_ACTIONS:
                with open(os.environ['GITHUB_OUTPUT'], 'a') as f:
                    f.write(f"error={str(e)[:100]}\n")

# ============================================================================
# GITHUB ACTIONS WORKFLOW GENERATOR
# ============================================================================
def generate_github_workflow():
    """Generate GitHub Actions workflow file"""
    workflow_content = """name: Crypto Trading Bot

on:
  schedule:
    # Run every 4 hours
    - cron: '0 */4 * * *'
  push:
    branches: [ main ]
  workflow_dispatch:  # Manual trigger

jobs:
  run-bot:
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout repository
      uses: actions/checkout@v4
      
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
        
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        
    - name: Run Trading Bot
      env:
        COINAPI_KEY: ${{ secrets.COINAPI_KEY }}
        GITHUB_ACTIONS: 'true'
      run: |
        python crypto_bot.py
        
    - name: Upload results
      uses: actions/upload-artifact@v3
      with:
        name: trading-results
        path: results/
        retention-days: 7
        
    - name: Generate Summary
      run: |
        echo "## Trading Bot Results" >> $GITHUB_STEP_SUMMARY
        echo "Completed at: $(date)" >> $GITHUB_STEP_SUMMARY
        echo "" >> $GITHUB_STEP_SUMMARY
        if [ -f results/trading_report.json ]; then
          echo "### Summary" >> $GITHUB_STEP_SUMMARY
          cat results/trading_report.json | python3 -c "import json,sys; data=json.load(sys.stdin); print(f'Processed {len(data[\"predictions\"])} coins'); print(f'Buy signals: {len(data[\"summary\"][\"buy_signals\"])}'); print(f'Sell signals: {len(data[\"summary\"][\"sell_signals\"])}')" >> $GITHUB_STEP_SUMMARY
        fi
"""
    
    # Create .github/workflows directory
    os.makedirs('.github/workflows', exist_ok=True)
    
    # Write workflow file
    with open('.github/workflows/crypto_bot.yml', 'w') as f:
        f.write(workflow_content)
    
    print("Generated GitHub Actions workflow at .github/workflows/crypto_bot.yml")

# ============================================================================
# REQUIREMENTS GENERATOR
# ============================================================================
def generate_requirements():
    """Generate requirements.txt file"""
    requirements = """numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
xgboost>=1.5.0
tensorflow>=2.10.0  # or tensorflow-cpu for lighter version
requests>=2.28.0
python-dotenv>=0.20.0
sqlalchemy>=1.4.0
"""
    
    with open('requirements.txt', 'w') as f:
        f.write(requirements)
    
    print("Generated requirements.txt")

# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    """Main entry point"""
    
    # Generate supporting files
    print("Generating supporting files...")
    generate_requirements()
    generate_github_workflow()
    
    # Create .env template if not exists
    if not os.path.exists('.env'):
        with open('.env', 'w') as f:
            f.write("""# Crypto Trading Bot Configuration
COINAPI_KEY=your_api_key_here
COINGECKO_API_KEY=optional_api_key_here
""")
        print("Generated .env template")
    
    # Initialize configuration
    config = Config()
    
    # Check for API key
    if config.COINAPI_KEY == 'YOUR_API_KEY_HERE':
        print("\n⚠️  WARNING: No API key found!")
        print("Please set your CoinAPI key:")
        print("1. In .env file: COINAPI_KEY=your_key")
        print("2. Or as environment variable")
        print("3. Or in GitHub Secrets if using GitHub Actions")
        
        # Try to get from user input
        try:
            api_key = input("\nEnter your CoinAPI key (or press Enter to use demo mode): ").strip()
            if api_key:
                config.COINAPI_KEY = api_key
                print("Using provided API key")
            else:
                print("Using demo mode with limited functionality")
        except:
            print("Using demo mode with limited functionality")
    
    # Initialize and run bot
    print("\nInitializing Crypto Trading Bot...")
    bot = CryptoTradingBot(config)
    
    print("\nStarting bot execution...")
    bot.run()
    
    print("\n" + "="*60)
    print("SETUP INSTRUCTIONS FOR GITHUB ACTIONS:")
    print("="*60)
    print("1. Push this code to a GitHub repository")
    print("2. Add your CoinAPI key as a secret:")
    print("   Settings > Secrets and variables > Actions > New repository secret")
    print("   Name: COINAPI_KEY")
    print("   Value: your_api_key")
    print("3. The workflow will run automatically every 4 hours")
    print("4. Results will be available as artifacts")
    print("\nFor local testing, run: python crypto_bot.py")

# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    # Check if running in GitHub Actions
    if os.getenv('GITHUB_ACTIONS'):
        print("Running in GitHub Actions environment")
    
    # Run main function
    main()
