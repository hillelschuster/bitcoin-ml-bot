# 📈 ML Models — Bitcoin Trading Bot

This directory contains the trained machine learning models and supporting files used by the Bitcoin ML Trading Bot.

## Contents

### ✅ `lstm_model.h5`
- Trained LSTM model used for time-series prediction
- Learns sequential BTC/USDT price and volume patterns

### ✅ `lstm_scaler.pkl`
- Feature scaler for normalizing input data
- Must be applied before feeding data to the model

### 📁 `history/`
- Stores archived models and training metrics
- Enables rollback, version comparison, and performance tracking

---

## 🔁 Model Lifecycle

The model is managed automatically by the bot:

1. **Prediction** — `src/models/model_lstm_core.py` handles signal generation using the LSTM model.
2. **Monitoring** — `src/monitoring/model_monitor.py` checks prediction accuracy and triggers retraining if needed.
3. **Retraining** — `src/models/model_retrain_loop.py` retrains the model using recent market data.
4. **Evaluation** — `src/models/model_evaluator.py` compares the new model with the current one and replaces it if better.

All files support **hot-reload**, **versioning**, and **self-healing logic**.
