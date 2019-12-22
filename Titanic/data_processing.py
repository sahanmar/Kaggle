import csv

from pathlib import Path
from typing import Dict, List


def get_data(file_path: Path) -> List[List[str]]:
    with open(file_path, newline="") as file:
        data = [row for row in csv.reader(file, delimiter=",")]
    return data


def create_dict(data: List[List[str]]) -> Dict[str, List[str]]:
    dict_data: Dict[str, List[str]] = {}
    header = data[0]
    for index, value in enumerate(header):
        dict_data[value] = [row[index] for row in data[1:]]
    return dict_data


def main():

    ### Define Path and check if its valid ###
    data_path = Path("Titanic/data/train.csv")  # Using of PathLib
    if data_path.is_file() is False or data_path.suffix != ".csv":
        raise Exception("this is not file path or wrong format...")

    ### Data Processing ###
    data = get_data(data_path)
    titanic_dict = create_dict(data)


if __name__ == "__main__":
    main()
