# CFD-ML GUI for Fluidized Bed Reactor

## Overview

This repository contains a Python-based graphical user interface (GUI) for exploring predictions from an artificial neural network (ANN) surrogate model developed for fluidized bed reactor simulations.

The GUI uses a trained ANN model and associated preprocessing scalers to generate predictions without requiring a new CFD simulation for each query.

## Features

* ANN-based surrogate prediction
* One-dimensional profile visualization
* Two-dimensional contour visualization
* Axial and radial analysis
* Export of prediction results and plots, where supported by the application

## Model Files

The `model` directory contains the trained ANN model and the associated scaler files required to run the GUI.

* `final_tuned_ann_3output.h5` — trained ANN model
* `scaler_X.pkl` — input scaler
* `scaler_y.pkl` — output scaler

## Requirements

Python and the required libraries must be installed. See `requirements.txt` for the dependency list once it has been added to this repository.

## Running the Application

Run the following command from the repository directory:

```bash
python app.py
```

## Intended Use

This software is intended for research and educational use in CFD-based fluidized bed reactor modelling and ANN surrogate prediction.

## Citation

If you use this software in your research, please cite the associated publication when available. A permanent software DOI may be added in a future release.

## Software Availability

Source code and trained model files are available in this public GitHub repository.

Repository: https://github.com/Rubina-Ak/CFD-ML-GUI-Fluidized-Bed
