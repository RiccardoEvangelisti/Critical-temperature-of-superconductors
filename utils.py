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
        if parameters != {}:
            parameters_rebuild = dict()
            for param in parameters.items():
                name_param = tag + "__" + param[0]
                value_param = param[1] if type(param[1]) is list else [param[1]]
                parameters_rebuild.update({name_param: value_param})
        else:
            parameters_rebuild = parameters
        self.parameters = parameters_rebuild


class Pipe:
    """
    Class to represent a Pipeline, in which are calculated all possible combinations of the parameters.
    (All combinations of parameters share the same Pipeline object)

    ## Parameters
    tag:
        Name of the Pipeline
    pipe:
        Actual Pipeline object
    parameter_grid:
        List of dictionaries (all possible combinations)
    """

    def __init__(self, *steps: Step):
        name_rebuild = ""
        constructors = list()
        parameters = dict()
        for index, step in enumerate(steps):
            if index == 0:
                name_rebuild = step.tag
            else:
                name_rebuild = name_rebuild + " + " + step.tag
            constructors.append((step.tag, step.constructor))
            parameters.update(**step.parameters)
        self.tag = name_rebuild
        self.pipeline = Pipeline(constructors)
        keys, values = zip(*parameters.items()) if parameters != {} else ([], [])
        self.parameter_grid = [{k: [v] for k, v in zip(keys, v)} for v in itertools.product(*values)]


class MultiplePipes:
    """
    Class to represent a list of Pipelines, in which are calculated all possible combinations of the parameters.

    ## Parameters
    combinations = (pipeline, parameters, tag):
        pipeline:
            Actual Pipeline
        parameters:
            Dictionary of parameters
        tag:
            Name of the Pipeline

    """

    def __init__(self, *pipes: Pipe):
        self.combinations = []
        for pipe in pipes:
            for parameters in pipe.parameter_grid:
                self.combinations.append((pipe.pipeline, parameters, pipe.tag))


def combination_already_tested(file_name, parameters, tag):
    """
    Check if the combination of given parameters is already tested
    """
    if os.path.isfile(file_name):
        outputs = pd.read_csv(file_name)
        # Re-create the parameters dictionary to be comparable with the outputs dataset
        parameters = pd.DataFrame({**parameters, "tag": [tag]}).fillna("NaN")
        # if all parameters are already present in the output:
        if all(x in outputs.columns for x in parameters.keys()):
            # take only the same columns
            outputs = outputs.loc[:, parameters.keys()].replace(np.nan, "NaN")
            # find if actual parameters are already present in the output dataset
            if (outputs == parameters.loc[0]).all(axis=1).any():
                return True
    return False


def print_results(file_name, head):
    df = pd.read_csv(file_name)
    return (
        df[["tag", "R2", "MSE"] + [col for col in df.columns if col not in ["tag", "R2", "MSE"]]]
        .sort_values(by="R2", ascending=False)
        .head(head)
        .style.format(precision=4)
    )
