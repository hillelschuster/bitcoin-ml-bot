# 🧪 Test Suite Overview

This folder contains all unit, integration, and diagnostic tests for the modular AI trading bot system. The structure is organized for clarity, scalability, and modularity.

---

## 📂 Directory Structure

### 🔧 Root-Level Tests (Functional Modules)
These test files directly correspond to core logic modules and utilities in the bot system:

- `test_order_manager.py` – Tests trade execution, entry/exit logic
- `test_price_utils.py` – Tests price validation and price fetching logic
- `test_state_manager.py` – Tests file-based state management and file locks
- `test_trade_logger.py` – Tests logging of trades and summaries

Each test here targets a **specific utility or service module** with focused unit tests.

---

### 📁 `diagnostics/` – Integration & Behavior Tests
This folder contains **diagnostic-style tests** that simulate higher-level workflows or monitor behavior across systems:

- `test_circuit_breaker.py` – Tests emergency exit trigger logic
- `test_close_position.py` – Tests logic for closing a position safely
- `test_model_monitor.py` – Tests model monitoring and alerting
- `test_exchange.py` – High-level test for unified Exchange interface

**Use cases for diagnostic tests:**
- Emergency position repair (e.g., `test_close_position.py` when bot crashes with open position)
- Safety logic validation (e.g., `test_circuit_breaker.py` to verify breaker activation)
- API connectivity checks (e.g., `test_exchange.py` to validate exchange connection)
- Model evaluation (e.g., `test_model_monitor.py` to compare model performance)

Use this folder to simulate **multi-component logic** and integration health checks.

---

### 📁 `core_utils/` – Pure Logic & Reasoning Tests
This folder is intended for **logic-only utility modules**:

- `test_reason_explainer.py` – Tests confidence-based reasoning, sentiment explanations, and model audit summaries

Keep this folder minimal — for modules that do not require exchange access, external dependencies, or full bot execution context.

---

## ✅ Best Practices

- Keep tests small and modular
- Use `unittest` for core logic; `pytest` also supported
- Avoid duplication across `diagnostics/` and root-level tests
- Use mocks when testing FastAPI, TensorFlow, or file I/O

---

## 📦 Notes

- You can run all tests using:
  ```bash
  python -m unittest discover tests/
  ```

Or with pytest:

```bash
pytest tests/ -v
```

## 🧠 Reminder
The diagnostic scripts are NOT used in production trading, but they are critical for:
- Regression testing after changes
- Emergency position repair
- Safety logic validation

Keep them maintained and run them after major updates or crashes.
