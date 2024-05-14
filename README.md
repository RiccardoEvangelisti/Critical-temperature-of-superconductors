# Critical Temperature of Superconductors

Kaggle Competition available [here](https://www.kaggle.com/competitions/critical-temperature-of-superconductors/overview).

### Description
The phenomenon of superconductivity ([Wikipedia](https://en.wikipedia.org/wiki/Superconductivity)) was discovered by Heike Kamerlingh-Onnes in 1911.

Superconductivity is a property of certain substances and materials whose *electrical resistance* drops to zero when the temperature drops to a certain level. Superconductivity occurs in a spike when the temperature drops. The temperature at which the spike occurs is called the **critical temperature**.

Although superconductivity was discovered more than a century ago, many of its properties are poorly understood, such as the relationship between superconductivity and the chemical/structural properties of materials.


### Objectives
Develop machine learning algorithms that allow to establish a relationship between the chemical composition, various properties of superconductors, and their critical temperature.


### (1) Dataset Description
The dataset comes from a database of superconducting materials compiled by Japan's National Institute of Materials Science (NIMS).

See [0_Data_Exploration.ipynb](https://github.com/RiccardoEvangelisti/Evangelisti-Critical-temperature-of-superconductors/blob/b2697efe428f7c333176c3062dc0e6a3e26a957b/0_Data_Exploration.ipynb)

### (2) Models Training
Different models are trained:
    - Linear Regression
    - Random Forest
    - XGBoost
    - KNN
    - SVM
Using several preprocessing configurations and combinations:
    - StandardScaler, MinMaxScaler
    - Normalizer L1, L2, Max
    - PCA
    - Train only on Properties or Formula dataset
    - Removing high correlated features

See [1_Training.ipynb](https://github.com/RiccardoEvangelisti/Evangelisti-Critical-temperature-of-superconductors/blob/cb8b27920a9046c159abe8b43eea98b0773ce812/1_Training.ipynb)

### (3) Explainability


### Results
