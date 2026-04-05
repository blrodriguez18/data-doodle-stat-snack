# Founder Dashboard Starter

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

## What it does
- Upload a CSV or load sample data
- Shows a data preview
- Auto-detects date and numeric columns
- Builds charts, a correlation matrix, and missing-value summary
- Flags anomalies on a time series
- Forecasts the next few periods with a simple trend model
- Produces ML model insights

## Best CSV shape
A CSV works best if it has:
- one date column
- one or more numeric columns like revenue, signups, churn, users

## Next upgrade
- Include ML model comparisons
- Add OpenAI-generated insights
