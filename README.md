# Critical Temperature of Superconductors

Kaggle Competition available [here](https://www.kaggle.com/competitions/critical-temperature-of-superconductors/overview).

In the [one-page/](https://github.com/RiccardoEvangelisti/Evangelisti-Critical-temperature-of-superconductors/tree/main/one-page) folder are available the one-page versions of the whole project, in HTML format (with *hiplot tool* interaction enabled) and PDF format (with *hiplot tool* interaction NOT enabled)

## Problem Description
The phenomenon of superconductivity ([Wikipedia](https://en.wikipedia.org/wiki/Superconductivity)) was discovered by Heike Kamerlingh-Onnes in 1911.

Superconductivity is a property of certain substances and materials whose electrical resistance drops to zero when the temperature equals to a certain value, called the **critical temperature**.

Many of the superconductivity properties are poorly understood, especially if the critical temperature can be predicted from the chemical and physical properties of the material.


## Objectives
1. Develop ML algorithms that can correctly predict the critical temperature, given the chemical structure and physical properties of a substance
2. Find which features are the most relevant in the estimation


## 1. Dataset Description
The dataset comes from a database of superconducting materials compiled by Japan's National Institute of Materials Science (NIMS).

See [0_Data_Exploration](https://github.com/RiccardoEvangelisti/Evangelisti-Critical-temperature-of-superconductors/blob/main/0_Data_Exploration.ipynb) notebook.

## 2. Models Training
Different models are trained:
- Linear Regression
- Random Forest
- XGBoost
- KNN
- SVM

Using several preprocessing configurations and combinations:
- Removing highly correlated features
- StandardScaler, MinMaxScaler
- Normalizer L1, L2, Max
- PCA
- Train only on Properties or Formula dataset

See [1_Training](https://github.com/RiccardoEvangelisti/Evangelisti-Critical-temperature-of-superconductors/blob/main/1_Training.ipynb) notebook.

## 3. Relationship between Critical Temperature and other features
To investigate on the relationship between critical temperature and other features, have been considered the following indicators:
- the *coefficients* of the Linear Regression model
- the *feature importance* based on *mean decrease in impurity*, of Random Forest and XGBoost models
- the *feature importance* based on *feature permutation*, of Random Forest and XGBoost models

See [2_Features_Importance](https://github.com/RiccardoEvangelisti/Evangelisti-Critical-temperature-of-superconductors/blob/main/2_Features_Importance.ipynb) notebook.

## 4. Best Result

|   |   |
|---|---|
|Best Model| **XGBoost**|
|Preprocessing| **None**|
| MSE | **78.09** |
| R^2| **0.931** |

Mainly looking at the *Feature Permutation* of the XGBoost model, the most "important" features are: ***Cu, Ca, Ba, O, range_ThermalConductivity, Valence***

See [3_Results](https://github.com/RiccardoEvangelisti/Evangelisti-Critical-temperature-of-superconductors/blob/main/3_Results.ipynb) notebook.