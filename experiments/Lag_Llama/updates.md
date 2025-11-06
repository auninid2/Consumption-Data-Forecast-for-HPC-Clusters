# Lag-Llama Model Execution and Forecast Analysis

This Jupyter Notebook demonstrates the use of the **Lag-Llama** model for time series forecasting on energy data. It includes environment setup, data loading, prediction generation, and forecast performance analysis.

---

## 1. Environment Setup and Data Loading

* Cloned the **lag-llama** repository and installed the required packages.
* Upgraded `torch` and `torchvision`.
* Created dummy `gluonts` modules to avoid import errors with the pre-trained model.
* Verified PyTorch version: **2.7.1+cu126**.
* **CUDA not available**: Computations were run on CPU.

### Data:

* Preprocessed CSVs: `train_data.csv`, `val_data.csv`, `test_data.csv`.
* Metadata from `metadata.json`:

  * **Prediction Length**: 24 hours
  * **Context Length**: 504 hours
  * **Frequency**: Hourly (`H`)
  * **Input Series**:

    * `'Erdgas', 'Sonstige Konventionelle', 'Biomasse', 'Braunkohle', 'Pumpspeicher', 'Steinkohle', 'Sonstige Erneuerbare', 'Photovoltaik', 'Wind Offshore', 'Wind Onshore', 'carbonIntensity', 'Wasserkraft', 'renewable_percentage', 'co2_intensity'`
  * **Primary Target**: `carbonIntensity`

### Dataset Sizes:

* Train: 3067 entries
* Validation: 657 entries
* Test: 658 entries

---

## 2. GluonTS Dataset Creation

* Created GluonTS datasets for `carbonIntensity`.
* Fixed `input_size` mismatch in `LagLlamaEstimator` to align with the pre-trained model (univariate: `input_size = 1`).

---

## 3. Lag-Llama Model Prediction

* Ran prediction on the `test_dataset` using pre-trained model.
* Device: **CPU**

### Configuration:

* Samples: **20**
* Prediction Length: **24 hours**
* Context Length: **504 hours**
* RoPE scaling: **Not used**

**Result:** Forecast shape = (20 samples, 24 prediction length)

---

## 4. Forecast Visualization

* Forecast plot for `carbonIntensity` includes:

  * Actual values
  * Forecast mean
  * 80% confidence interval
  * Date/time formatted x-axis

---

## 5. Performance Metrics Analysis

Metrics for `carbonIntensity` forecast:

* **MAE**: 19.349
* **RMSE**: 25.105
* **MAPE**: 9.4%

---

## 6. Energy Mix & CO2 Emissions Prediction Analysis

### Renewable Energy Mix

* `renewable_percentage` forecast **not available** for detailed analysis (not a primary target).
* Likely included as a **dynamic feature**.

### CO2 Intensity (`carbonIntensity`):

* **Start of forecast**: 232.75 kg CO2/MWh
* **End of 24h forecast**: 204.06 kg CO2/MWh
* **Change**: −28.69 kg CO2/MWh (**−12.3%**)

### Energy Price

* No forecast available (not a primary target).

---

## 7. Advanced Pattern Analysis

### Correlation Matrix:

* Not computed due to insufficient primary targets.

### Hourly Trends:

* Plot available for `carbonIntensity` over 24-hour forecast.
* `renewable_percentage` skipped (not a primary target).

### Energy Transition Trend:

* Renewable energy expected to **decrease by 0.548 percentage points per hour**.
* Indicates **increased reliance on conventional energy sources**.

---

## Summary of Key Findings

* Successfully forecasted 24-hour `carbonIntensity` using Lag-Llama.
* Predicted **12.3% decrease** in `carbonIntensity` over forecast horizon.
* Performance metrics:

  * **MAE**: 19.349
  * **RMSE**: 25.105
  * **MAPE**: 9.4%
* **Renewable energy share** expected to decline.
* **Energy price** not analyzed due to exclusion as a primary target.
