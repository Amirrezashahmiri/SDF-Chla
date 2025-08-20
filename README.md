# Interpretable, Spatiotemporal Deep Learning Framework for Chlorophyll-a Forecasting

This repository contains the official source code and data for the research paper titled: *"Interpretable, Spatiotemporal Deep Learning Framework for Chlorophyll-a Forecasting from Fused Satellite and Multidimensional Earth System Data"*.

---

## Abstract
Algal blooms are intensifying as sea temperatures rise and nutrient inputs expand, making the forecasting of Chlorophyll-a (Chl-a) concentrations critical for early warning. This research introduces a deep learning framework that fuses satellite observations (MODIS-Aqua) with multidimensional Earth system data (ERA5, HYCOM, ETOPO1) to deliver reliable next-month Chl-a forecasts at a 9 km resolution. By leveraging novel transformed spatiotemporal features and a physics-informed approach, our Deep Neural Network (DNN) model achieves high accuracy (R² = 0.84, RMSE = 1.03 mg/m³). Critically, we employ Explainable AI (XAI) techniques, including SHAP and Integrated Gradients, to provide interpretable insights into the key physical drivers (e.g., bathymetry, SST fluctuations) of algal bloom dynamics, transforming the model from a "black box" into a diagnostic tool for sustainable coastal management.

---

## Repository Structure
This repository is organized to ensure the reproducibility of our results. The key files are:

-   `SST_Transformation.py`: Python script to calculate the novel spatiotemporal features (e.g., SSGF, TSGF) from the primary datasets. This is the first step in the feature engineering process.
-   `RF_SHAP.py`: Python script to train the Random Forest (RF) model and perform the SHAP analysis for feature interpretation.
-   `DNN_IG.py`: Python script to train the Deep Neural Network (DNN) model and perform the Integrated Gradients (IG) analysis for feature interpretation.
-   `Processed_Dataset.xlsx`: A sample of the final, processed dataset used as input for training and validating the models. This file shows the structure of the data after all preprocessing and feature engineering steps.
-   `DNN_Predictions.xlsx` & `RF_Predictions.csv`: The prediction outputs from the DNN and RF models for the out-of-sample test set.
-   `AQ*.TXT`: Raw or intermediate data files derived from MODIS-Aqua observations for the study period.
-   `README.md`: This file, providing an overview and guide to the repository.

---

## Data Sources
Our framework integrates several publicly available datasets. The raw data can be accessed from their original sources:

* **MODIS-Aqua Ocean Color (NASA OBPG):** Level-3 monthly Chl-a and SST data. [Link](https://oceancolor.gsfc.nasa.gov/)
* **ECMWF Reanalysis v5 (ERA5):** Meteorological data (wind, precipitation). [Link](https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5)
* **Hybrid Coordinate Ocean Model (HYCOM):** Oceanic data (vertical salinity and temperature profiles). [Link](https://www.hycom.org/)
* **ETOPO1 Global Relief Model (NOAA):** Bathymetric data. [Link](https://www.ngdc.noaa.gov/mgg/global/)
