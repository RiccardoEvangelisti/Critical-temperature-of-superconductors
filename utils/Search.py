import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from utils.Combinations import Combination


def combination_already_tested(file_name: str, combination: Combination):
    """
    Check if the given Combination is already tested (i.e. present in the output file)
    """
    if os.path.isfile(file_name):
        outputs = pd.read_csv(file_name)
        combination: pd.DataFrame = combination.as_comparable()
        # if all parameters of the combination are already present in the output:
        if all(x in outputs.columns for x in combination.columns):
            # find if actual parameters are already present in the output dataset
            outputs = pd.concat([outputs, combination], axis=0).reset_index(drop=True).drop(columns=["MSE", "R2"])
            duplicated = outputs.duplicated(keep=False).any()
            return duplicated
    return False


def grid_search(
    OUTPUT_FOLDER: str,
    X_train,
    y_train,
    X_test,
    y_test,
    combinations: list[Combination],
    estimator_tag: str,
    save_results=True,
):
    """
    For each given Combination:
        - Check if the combination is already been tested (present in the output dataset)
        - If not: start the search with the Pipeline and the parameters of the Configuration instance, with a 3-folds Cross-Validation
        - Save the configuration with the MSE and R2 metrics, comparing the target predictions and test values

    ## Returns
    The last executed (best) estimator. Useful when is tested only one combination.
    """

    # Iterate over *all* combinations
    for index, combination in enumerate(combinations):
        print("\nCombination {}/{}  |  {}".format(index + 1, len(combinations), combination.tag))

        # Check if this combination is already tested
        if save_results:
            file_name = OUTPUT_FOLDER + estimator_tag + "_output.csv"
            if combination_already_tested(file_name, combination):
                print("  ==> Already done. Skipped.")
                continue

        gs = GridSearchCV(
            estimator=combination.pipeline,
            param_grid=combination.parameters,
            n_jobs=-1,
            cv=3,
            verbose=0,
        )

        # Fit
        gs.fit(X_train, np.ravel(y_train))
        # Predict
        y_pred = gs.predict(X_test)
        # Test scores
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print("  ==> R2: {}\tMSE: {}".format(r2, mse))

        # Save results
        if save_results:
            results = combination.set_MSE(mse).set_R2(r2).as_df()
            if os.path.isfile(file_name):
                outputs = pd.read_csv(file_name)
                if not outputs.empty:
                    results = pd.concat([outputs, results], axis=0)
            results.to_csv(file_name, index=False)

    if save_results == False:
        return gs.best_estimator_


def best_hyperparameters(file_name, percentage):
    """
    Returns the hyperparameters of the best tested configurations (ordered by R2 metric).

    ## Parameters
    file_name:
        Name of the file containing the output
    percentage:
        Percentage of the total configurations from which the hyperparameters are taken
    """
    df = pd.read_csv(file_name)
    samples = df.shape[0] * percentage // 100
    df = df.sort_values(by="R2", ascending=False).drop(columns=["R2", "MSE"]).iloc[:samples]
    results = dict()
    for hyperparameter in df.columns:
        results.update({hyperparameter: list(df[hyperparameter].unique())})
    return results


def print_results(file_name, head):
    """
    Print the best hyperparameters (the first "head" configurations)
    """
    df = pd.read_csv(file_name)
    return (
        df[["tag", "R2", "MSE"] + [col for col in df.columns if col not in ["tag", "R2", "MSE"]]]
        .sort_values(by="R2", ascending=False)
        .head(head)
        .style.format(precision=4)
        .set_caption("R2 sorted")
    )
