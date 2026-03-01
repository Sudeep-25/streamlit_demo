# 🛡 Network IDS — Streamlit App

Random Forest-based Network Intrusion Detection System  
Built for Streamlit Cloud deployment.

## Files Required
```
├── app.py              ← Main Streamlit application
├── model.h5            ← Trained Random Forest model
├── dataset.csv         ← KDD CUP 99 dataset (for dashboard stats)
├── requirements.txt    ← Python dependencies
└── .streamlit/
    └── config.toml     ← Dark theme configuration
```

## Local Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Login
- Username: `admin`
- Password: `admin`
