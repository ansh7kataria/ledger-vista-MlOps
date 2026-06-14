---
title: Tourism Package Predictor
emoji: 🌍
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.40.0
app_file: app.py
pinned: false
---

# Wellness Tourism Package — Purchase Predictor

A Streamlit app that predicts whether a customer is likely to buy the Wellness
Tourism Package, so the sales team can focus follow-ups where they convert.

The model (XGBoost, selected on F1 over a RandomForest baseline) is loaded from
the Hugging Face model hub at `iamanshkataria/tourism-package-model`.
