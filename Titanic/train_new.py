import csv

from pathlib import Path
from typing import Dict, List


def get_data(file_path: Path) -> List[List[str]]: #function annotations
    with open(file_path, newline="") as file:
        data = [row for row in csv.reader(file, delimiter=",")] #generator
    return data


def create_dict(data: List[List[str]]) -> Dict[str, List[str]]:
    dict_data: Dict[str, List[str]] = {}
    header = data[0]
    for index, value in enumerate(header): #(with) index is 0-11, value is header elements
        dict_data[value] = [i[index] for i in data[1:]] #make key value from header =
        # = [element with index from list, which is element in data, from raw 1 to end including last element
    return dict_data,header


def main():
    ### Define Path and check if its valid ###
    data_path = Path("data/train.csv")  # Using of PathLib
    if data_path.is_file() is False or data_path.suffix != ".csv":
        raise Exception("this is not file path or wrong format...")
    ### Data Processing ###
    data = get_data(data_path)
    titanic_dict, header = create_dict(data)
    print(titanic_dict) #[header[0]]

if __name__ != "__main__": #mainly... for importing and checking :)
    main()
