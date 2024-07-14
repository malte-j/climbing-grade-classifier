import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    accuracy_score,
)
from sklearn.pipeline import Pipeline
from tsfresh.transformers import RelevantFeatureAugmenter
from get_processed_data import get_train_test_split
import pickle
import os
# import time
# from collections.abc import Iterator
# from contextlib import contextmanager


# @contextmanager
# def time_it() -> Iterator[None]:
#     tic: float = time.perf_counter()
#     try:
#         yield
#     finally:
#         toc: float = time.perf_counter()
#         print(f"Computation time = {1000*(toc - tic):.3f}ms")


def getSplits(window_size, overlap):
    print(f"Generating splits for: Window size: {window_size}, Overlap: {overlap}")
    train_test_split = get_train_test_split("malte", window_size, overlap)

    ts_train = []
    y_train = pd.Series()  # 0 = not hard, 1 = hard
    i = 0

    ts_test = []
    y_test = pd.Series()  # 0 = not hard, 1 = hard

    for split_i, split in enumerate(train_test_split):
        for difficulty, windows in split.items():
            for window in windows:
                data = pd.DataFrame(window)
                data = data.drop("PacketCounter", axis=1)
                data["id"] = i
                if split_i == 0:
                    ts_train.append(data)
                    y_train[i] = difficulty == "hard"
                else:
                    ts_test.append(data)
                    y_test[i] = difficulty == "hard"
                i += 1

    df_ts_train = pd.concat(ts_train)
    df_ts_test = pd.concat(ts_test)

    X_test = pd.DataFrame(index=y_test.index)
    X_train = pd.DataFrame(index=y_train.index)

    return (
        X_train,
        df_ts_train,
        y_train,
        X_test,
        df_ts_test,
        y_test,
    )


def runPipeline(X_train, df_ts_train, y_train, X_test, df_ts_test, y_test, config):
    print("Running pipeline")
    ppl = Pipeline(
        [
            (
                "augmenter",
                RelevantFeatureAugmenter(
                    column_id="id",
                    column_sort="SampleTimeFine",
                    column_kind=None,
                    column_value=None,
                ),
            ),
            ("classifier", RandomForestClassifier()),
        ]
    )
    ppl.set_params(augmenter__timeseries_container=df_ts_train)
    ppl.fit(X_train, y_train)

    # create folder if it does not exist
    try:
        os.makedirs("models/rf")
    except FileExistsError:
        pass

    with open(f"models/rf/ws{config['window_size']}_o{config['overlap']}", "wb") as f:
        pickle.dump(ppl, f)

    ppl.set_params(augmenter__timeseries_container=df_ts_test)
    y_pred = ppl.predict(X_test)

    return (
        classification_report(y_test, y_pred)
        + f"\nBalanced accuracy score: {balanced_accuracy_score(y_test, y_pred)}"
        + f"\nAccuracy Score: {accuracy_score(y_test, y_pred)}"
    )


def main():
    runsConfig = [
        {
            "window_size": 1 * 60,  # 1 second
            "overlap": 0,
        },
        {
            "window_size": 5 * 60,
            "overlap": 0,
        },
        {
            "window_size": 5 * 60,
            "overlap": 2 * 60,
        },
        {
            "window_size": 10 * 60,
            "overlap": 0,
        },
        {
            "window_size": 10 * 60,
            "overlap": 2 * 60,
        },
        {
            "window_size": 10 * 60,
            "overlap": 5 * 60,
        },
        {
            "window_size": 15 * 60,
            "overlap": 0,
        },
        {
            "window_size": 15 * 60,
            "overlap": 2 * 60,
        },
        {
            "window_size": 15 * 60,
            "overlap": 5 * 60,
        },
        {
            "window_size": 15 * 60,
            "overlap": 10 * 60,
        },
    ]

    for config in runsConfig:
        data = getSplits(config["window_size"], config["overlap"])
        pipelineResult = runPipeline(*data, config)
        print(pipelineResult)

        # print to file
        with open("random_forest_results_unbalanced.txt", "a") as f:
            f.write(
                f"Window size: {config['window_size']}, Overlap: {config['overlap']}\n"
            )
            f.write(pipelineResult)
            f.write("\n\n")


if __name__ == "__main__":
    main()
