# LSTM Model Usage Instructions

## Quick Start
```python
from tensorflow.keras.models import load_model
import pickle
import json

model = load_model('artifacts/lstm_model.keras')
with open('artifacts/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('artifacts/feature_columns.json', 'r') as f:
    features = json.load(f)
```

## Performance
- RMSE: 6.2552
- Total improvement: 13.28%
