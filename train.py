import csv
import json

def info_person(filename):
    info_arr = []
    with open(filename, newline='') as File:
        reader = csv.reader(File, delimiter =',')
        for row in reader:
            info_arr.append(row)

    info_dict = {}
    for i in range(1,len(info_arr)):
        person_arr = []
        for j in range(1,len(info_arr[0])):
            if info_arr[i][j] == "":
                info_arr[i][j] = "no info"
            person_arr.append(info_arr[i][j])
            info_dict[info_arr[i][0]] = person_arr
    print(info_dict)
    info_json = json.dumps(info_dict)
    print(info_json)

info_person('train.csv')

# Read by rows
# def csv_reader(filename):
#     reader = csv.reader(filename)
#     for row in reader:
#         print(row)

# Dictionary
# info = []
# csv_path = "cipy.csv"
# with open(csv_path, newline='') as csvFile:
#     reader = csv.DictReader(csvFile)
#     for row in reader:
#         info.append(row)
#     print(info)
