import pandas as pd
from sklearn.pipeline import Pipeline
import itertools


class Step:
    """
    Class to represent a single step in a Pipeline (sklearn.pipeline.Pipeline).

    Each given parameter name is converted to the format needed by the Pipeline: "namestep__parametername"

    ## Parameters
    tag:
        Name of the step
    constructor:
        Instance of the step
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


class Pipe:
    """
    Class representing a sequence of Steps that need to be executed in the given order.

    ## Parameters
    steps: list of Step instances
    """

    def __init__(self, *steps: Step):
        self.steps = list(steps)


class Combination:
    """
    A single combination of transformers and parameters to be executed.

    ## Parameters
    tag:
        name of the combination
    pipeline:
        Pipeline instance
    parameters:
        dictionary of parameters. Each parameter has a single value to test

    ## Methods
    set_MSE:
        Set the MSE of the result
    set_R2:
        Set the R2 of the result
    as_df:
        Converts the combination, plus MSE and R2 results, into a Pandas DataFrame with only one line
    as_comparable:
        Returns the Combination as Pandas DataFrame, with the tag name and None parameters as 'None' string
    """

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
        """
        Converts the combination, plus MSE and R2 results and the tag, into a Pandas DataFrame with only one line.

        The "None" value of a parameter is written as 'None' string, while the None value means the parameter is not set.
        """
        if self.MSE == None or self.R2 == None:
            return ValueError("Please set MSE and R2 parameters")

        out = pd.DataFrame({"tag": self.tag, "MSE": self.MSE, "R2": self.R2}, index=[0])

        params_df = pd.DataFrame(self.parameters, index=[0]).fillna("'None'")

        if not params_df.empty:
            out = pd.concat([out, params_df], axis=1)
        return out

    def as_comparable(self) -> pd.DataFrame:
        """
        Returns the Combination as Pandas DataFrame, with the tag name and None parameters as 'None' string.

        It eases the comparison between this Combination and the other Combinations saved in the output files.
        """
        out = pd.DataFrame({"tag": self.tag}, index=[0])
        params_df = pd.DataFrame(self.parameters, index=[0]).fillna("'None'")
        if not params_df.empty:
            out = pd.concat([out, params_df], axis=1)
        return out


def extract_combinations(*pipes: Pipe):
    """
    Extracts all possible combinations, given a list of Pipe objects.

    Each returned Combination a single configuration of parameters.

    ## Returns
    list of Combination objects
    """
    all_combinations = list()
    # Iterate over all Pipe of Steps
    for pipe in pipes:
        tag = ""
        constructors = list()
        all_parameters = dict()
        # Iterate over all steps
        for index, step in enumerate(pipe.steps):
            if index == 0:
                tag = step.tag
            else:
                tag = tag + " + " + step.tag
            constructors.append((step.tag, step.constructor))
            all_parameters.update(**step.parameters)

        pipeline = Pipeline(constructors)  # Needs as input a list of pairs (step_name, step_constructor)

        # For each possible combination of parameters
        keys, values = zip(*all_parameters.items()) if all_parameters != {} else ([], [])
        for v in itertools.product(*values):
            single_parameter_combination = {k: [v] for k, v in zip(keys, v)}
            all_combinations.append(Combination(tag, pipeline, single_parameter_combination))

    return all_combinations
