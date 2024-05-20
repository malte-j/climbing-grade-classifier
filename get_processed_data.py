import numpy as np


def process_file(file_path, start_ts, end_ts):
    raw_data = np.genfromtxt(
        file_path,
        delimiter=",",
        skip_header=12,
        names=[
            "PacketCounter",
            "SampleTimeFine",
            "Quat_W",
            "Quat_X",
            "Quat_Y",
            "Quat_Z",
            "FreeAcc_X",
            "FreeAcc_Y",
            "FreeAcc_Z",
        ],
        dtype=(int, int, float, float, float, float, float, float, float),
    )

    # normalize start to 0
    raw_data["SampleTimeFine"] -= raw_data["SampleTimeFine"][0]

    # drop all rows where SampleTimeFine is smaller than start_ts and larger than end_ts
    data = raw_data[
        (raw_data["SampleTimeFine"] >= start_ts)
        & (raw_data["SampleTimeFine"] <= end_ts)
    ]

    # normalize again
    data["SampleTimeFine"] -= data["SampleTimeFine"][0]

    return data


processed_data_malte = {
    "3": [
        process_file("measurements/2024-04-30/malte-3.csv", 13848061, 41408943),
        process_file(
            "measurements/2024-05-03/malte-3-bildschirm_an.csv", 27207765, 49546441
        ),
    ],
    "4": [process_file("measurements/2024-05-03/malte-4.csv", 17343773, 98990743)],
    "5-": [
        process_file("measurements/2024-05-03/malte-5minus.csv", 15535808, 64060764)
    ],
    "6": [process_file("measurements/2024-05-03/malte-6.csv", 35368972, 121547587)],
}


processed_data_luis = {
    "5-": [process_file("measurements/2024-05-03/luis-5minus.csv", 7915785, 103630640)],
}
