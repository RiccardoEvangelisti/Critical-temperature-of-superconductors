import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
import itertools


class Step:
    """
    Class to represent a single step in a Pipeline.
    Each parameter name is converted to the format needed by the Pipeline:
        "namestep__parametername"

    ## Parameters
    tag:
        Name of the step
    constructor:
        Object of the step
    parameters:
        Dictionary of parameters
    """

    def __init__(self, tag: str, constructor, parameters: dict = {}):
        self.tag = tag
        self.constructor = constructor
        parameters_rebuild = dict()
        # Iterate over all parameters
        if parameters != {}:
            for param in parameters.items():
                # Convert to the format needed by the Pipeline object
                name_param = tag + "__" + param[0]  # "namestep__parametername"
                value_param = param[1] if type(param[1]) is list else [param[1]]  # [value]
                parameters_rebuild.update({name_param: value_param})
        self.parameters = parameters_rebuild


class Combination:
    def __init__(self, tag, pipeline, parameters):
        self.pipeline = pipeline
        self.tag = tag
        self.parameters = parameters

        self.MSE = None
        self.R2 = None

    def set_MSE(self, mse):
        self.MSE = mse
        return self

    def set_R2(self, r2):
        self.R2 = r2
        return self

    def as_df(self):
        if self.MSE == None or self.R2 == None:
            return ValueError("Please set MSE and R2 parameters")

        out = pd.DataFrame({"tag": self.tag, "MSE": self.MSE, "R2": self.R2}, index=[0])

        params_df = pd.DataFrame(self.parameters, index=[0]).fillna("'None'")

        if not params_df.empty:
            out = pd.concat([out, params_df], axis=1)
        return out

    def as_comparable(self) -> pd.DataFrame:
        out = pd.DataFrame({"tag": self.tag}, index=[0])
        params_df = pd.DataFrame(self.parameters, index=[0]).fillna("'None'")
        if not params_df.empty:
            out = pd.concat([out, params_df], axis=1)
        return out


def combination_already_tested(file_name: str, combination: Combination):
    """
    Check if the combination of given parameters is already tested
    """
    if os.path.isfile(file_name):
        outputs = pd.read_csv(file_name)
        combination: pd.DataFrame = combination.as_comparable()
        # if all parameters of the combination are already present in the output:
        if all(x in outputs.columns for x in combination.columns):
            # take only the same columns
            # outputs = outputs.loc[:, combination.columns]
            # find if actual parameters are already present in the output dataset
            outputs = pd.concat([outputs, combination], axis=0).reset_index(drop=True).drop(columns=["MSE", "R2"])
            duplicated = outputs.duplicated(keep=False).any()
            return duplicated
    return False


def print_results(file_name, head):
    df = pd.read_csv(file_name)
    return (
        df[["tag", "R2", "MSE"] + [col for col in df.columns if col not in ["tag", "R2", "MSE"]]]
        .sort_values(by="R2", ascending=False)
        .head(head)
        .style.format(precision=4)
    )
