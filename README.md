# Advanced Time Series Forecasting with Transformers

This project demonstrates a complete **end‑to‑end time series forecasting pipeline** using **deep learning (Transformer encoder–decoder)** along with classical baselines like **SARIMAX** and optional **Prophet**.

## 🌟 Key Features

* Generates **synthetic multivariate weekly time series** (8 series, trend + seasonality + noise).
* Performs **scaling, windowing, and dataset creation** for multi-step forecasting.
* Builds a **Transformer-based forecasting model** with positional encoding.
* Uses **encoder–decoder architecture** for predicting the next *HORIZON* (default 8 weeks).
* Computes performance metrics: **SMAPE, MASE, RMSE**.
* Includes **SARIMAX baseline** for comparison.
* Automatically saves:

  * Trained model (`.pth`)
  * Results (`.json`)
  * Example prediction plot
  * PDF report (using ReportLab)
  * Synthetic dataset CSV

## 📁 Output Directory Structure

All outputs are stored in the `tf_out/` folder:

```
synthetic_series.csv
transformer_model_logeshwaran s.pth
results_transformer.json
example_series_0.png
logeshwaran report.pdf
```

## 🚀 How to Run

```bash
python transformer_forecasting_logeshwaran_s.py
```

## 🛠️ Dependencies

* PyTorch
* NumPy, Pandas
* Scikit-learn
* Matplotlib
* Statsmodels (for SARIMAX)
* ReportLab (for PDF)

## 🙌 Author

**Logeshwaran S** — Time series forecasting using advanced deep learning models.
