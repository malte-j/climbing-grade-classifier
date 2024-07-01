from typing import List, Optional

import numpy as np
from sklearn.model_selection import train_test_split


def process_file(
    file_path: str, start_ts: int, end_ts: int, annotation: Optional[str] = None
) -> np.ndarray:
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


def generate_windows(
    data: np.ndarray, window_size: int, overlap: int, drop_unfilled_windows: bool
) -> List[np.ndarray]:
    windows = []
    for i in range(0, len(data), window_size - overlap):
        window = data[i : i + window_size]
        if not drop_unfilled_windows or len(window) == window_size:
            windows.append(window)
    return windows


def get_processed_data(
    person: str,
    grades: List[str],
    window_size: int = 0,
    overlap: int = 0,
    drop_unfilled_windows: bool = True,
):
    """
    Get processed data for a person and a list of grades.
    Parameters:
    - person: either "malte" or "luis"
    - grades: list of grades to get data for
    - window_size: window size in n samples to include, with a frequency of 60 Hz to get 30s you need to set it to 60*30 = 1800
    - overlap: overlap in samples
    - drop_unfilled_windows: if True, only windows with exactly window_size samples are included
    """
    if person == "malte":
        processed_data = processed_data_malte
    elif person == "luis":
        processed_data = processed_data_luis
    else:
        raise ValueError("Unknown person")

    data = []
    for grade in grades:
        if grade not in processed_data:
            continue
        for sample in processed_data[grade]:
            if window_size == 0:
                data.append(sample)
            else:
                data = data + generate_windows(
                    sample, window_size, overlap, drop_unfilled_windows
                )

    return data


processed_data_malte = {
    "3": [
        process_file("measurements/2024-04-30/malte-3.csv", 13848061, 41408943),
        process_file("measurements/2024-05-03/malte-3.csv", 27207765, 49546441),
        process_file(
            "measurements/2024-05-03/malte-3-bildschirm_aus.csv", 18316261, 46820050
        ),
        process_file("measurements/2024-05-26/malte-3_1.csv", 22927121, 62276359),
        process_file("measurements/2024-05-26/malte-3_2.csv", 35592329, 70026390),
        process_file("measurements/2024-05-26/malte-3_3.csv", 20767801, 47627539),
        process_file("measurements/2024-06-09/malte-3.csv", 38352361, 70034385),
        process_file("measurements/2024-06-13/malte-3.csv", 24345131, 60246498),
        process_file("measurements/2024-06-13/malte-3_2.csv", 26936350, 63899199),
    ],
    "4": [
        process_file("measurements/2024-04-30/malte-4.csv", 12833225, 50560803),
        process_file("measurements/2024-05-03/malte-4.csv", 17343773, 98990743),
        process_file("measurements/2024-06-13/malte-4.csv", 23562047, 55113000),
        process_file("measurements/2024-06-13/malte-4_2.csv", 31481811, 69008021),
    ],
    "5-": [
        process_file("measurements/2024-05-03/malte-5minus.csv", 15535808, 64060764)
    ],
    "5": [
        process_file("measurements/2024-06-13/malte-5.csv", 46554465, 115676828),
        process_file("measurements/2024-06-13/malte-5_2.csv", 35597328, 89274487),
    ],
    "6": [
        process_file("measurements/2024-05-03/malte-6.csv", 35368972, 121547587),
        # process_file(
        #     "measurements/2024-04-30/malte-6.csv",
        #     37257707,
        #     49672677,
        #     "Extreme average acceleration magnitude of 7, might be an outlier.",
        # ),
        process_file("measurements/2024-05-26/malte-6_1.csv", 35925844, 114430150),
        process_file("measurements/2024-05-26/malte-6_2.csv", 36491087, 114624550),
        process_file("measurements/2024-06-13/malte-6.csv", 41535055, 141851639),
    ],
    "6+": [
        process_file(
            "measurements/2024-05-26/malte-6p_1.csv",
            21328619,
            130908568,
            "This one was slightly weird because there was a clear end, but no clearly visible start.",
        ),
        process_file("measurements/2024-05-26/malte-6p_2.csv", 61063659, 159628039),
        process_file("measurements/2024-06-09/malte-6p.csv", 59272943, 161310032),
        process_file("measurements/2024-06-09/malte-6p_2.csv", 51777649, 170703368),
        process_file("measurements/2024-06-13/malte-6p.csv", 45585106, 185489448),
    ],
    "7": [
        process_file("measurements/2024-06-13/malte-7.csv", 32408679, 231538620),
    ],
}


processed_data_luis = {
    "3": [
        process_file("measurements/2024-04-30/luis-3.csv", 3306648, 45876383),
        process_file("measurements/2024-05-03/luis-3.csv", 15909563, 59147842),
        process_file("measurements/2024-06-09/luis-3.csv", 58395424, 106379845),
        process_file("measurements/2024-06-13/luis-3.csv", 22768958, 82762259),
        process_file("measurements/2024-06-13/luis-3_2.csv", 24225032, 43004217),
    ],
    "4": [
        process_file("measurements/2024-06-13/luis-4.csv", 59837548, 153137678),
        process_file("measurements/2024-06-13/luis-4_2.csv", 30895700, 135832938),
        process_file("measurements/2024-06-13/luis-4_3.csv", 24270977, 148639748),
    ],
    "5-": [process_file("measurements/2024-05-03/luis-5minus.csv", 7915785, 103630640)],
    "5": [
        process_file("measurements/2024-04-30/luis-5.csv", 18215188, 147090949),
        process_file("measurements/2024-06-09/luis-5.csv", 58106964, 340573344),
        process_file("measurements/2024-06-13/luis-5_2.csv", 73859300, 194768025),
    ],
}

# train, test = get_train_test_split()
# train["hard"] = List<ndarray>


def get_train_test_split(
    person: str,
    window_size: int = 0,
    overlap: int = 0,
    drop_unfilled_windows: bool = True,
    train_ratio: float = 0.9,
):
    if person == "malte":
        processed_data = processed_data_malte
    elif person == "luis":
        processed_data = processed_data_luis
    else:
        raise ValueError("Unknown person")

    difficulties = {
        "easy": processed_data["3"] + processed_data["4"],
        "hard": processed_data["6"] + processed_data["6+"],
    }

    train = {}
    test = {}

    for difficulty, samples in difficulties.items():
        train[difficulty], test[difficulty] = train_test_split(
            samples, train_size=train_ratio, shuffle=True, random_state=42
        )

    for split in [train, test]:
        for difficulty, samples in split.items():
            if window_size == 0:
                split[difficulty] = np.concatenate(samples)
            else:
                windows = []
                for sample in samples:
                    windows = windows + generate_windows(
                        sample, window_size, overlap, drop_unfilled_windows
                    )
                split[difficulty] = windows

    return train, test
