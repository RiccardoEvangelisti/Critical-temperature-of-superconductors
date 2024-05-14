import numpy as np
import pandas as pd


DATA_FOLDER = "data/"


# Remove Features with High Correlation
class HighCorrFeaturesRemover:
    """
    Removes features with high correlation, according to the 'corr_threshold' parameter.

    Class that provide the fit and transform methods, in order to be used as a "transformer" into the Pipeline class

    ## Parameters
    corr_threshold: float (0,1]
        Percentage of minimum correlation between features, above which a feature is removed
    """

    def fit(self, X, y=None):
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        self.cols_to_drop = [column for column in upper.columns if any(upper[column] >= self.corr_threshold)]
        return self

    def transform(self, X):
        # print("{} Cols Removed: {}".format(len(cols_to_drop), cols_to_drop))
        return X.drop(columns=self.cols_to_drop)

    def set_params(self, corr_threshold):
        self.corr_threshold = corr_threshold
        return self
    
    def get_feature_names_out(self):
        return 


# Keep only the Features of Properties dataset
class OnlyProperties:
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(
            columns=pd.read_csv(DATA_FOLDER + "formula_train.csv").drop(columns=["critical_temp", "material"]).columns
        )


# Keep only the Features of Formula dataset
class OnlyFormula:
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=pd.read_csv(DATA_FOLDER + "train.csv").drop(columns=["critical_temp"]).columns)
