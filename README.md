# Critical Temperature of Superconductors

Kaggle Competition available [here](https://www.kaggle.com/competitions/critical-temperature-of-superconductors/overview).

## Problem Description
The phenomenon of superconductivity ([Wikipedia](https://en.wikipedia.org/wiki/Superconductivity)) was discovered by Heike Kamerlingh-Onnes in 1911.

Superconductivity is a property of certain substances and materials whose *electrical resistance* drops to zero when the temperature drops to a certain level. Superconductivity occurs in a spike when the temperature drops. The temperature at which the spike occurs is called the **critical temperature**.

Although superconductivity was discovered more than a century ago, many of its properties are poorly understood, such as the relationship between superconductivity and the chemical/structural properties of materials.


## Objectives
Develop machine learning algorithms that allow to establish a relationship between the chemical composition, various properties of superconductors, and their critical temperature.


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
- Removing high correlated features
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

## 4. Results

|   |   |
|---|---|
|Best Model| **XGBoost**|
|Preprocessing| **None**|
| R^2| **0.931** |
| MSE | **78.09** |

Mainly looking at the *Feature Permutation* of the XGBoost model, the most "important" features are: ***Cu, Ca, Ba, O, range_ThermalConductivity, Valence***

See [3_Results](https://github.com/RiccardoEvangelisti/Evangelisti-Critical-temperature-of-superconductors/blob/main/3_Results.ipynb) notebook.