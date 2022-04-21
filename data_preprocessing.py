import random
import math
import re


def load_data(file_name, separator=","):
    rows = []
    with open(file_name, encoding='utf-8') as file:
        for row in file:

            if separator == ' ':
                row = re.split('\s+', row.strip())
            else:
                row = row.strip().split(separator)
            rows.append(row)
    return rows


# Replace missing value with most frequent value
def handle_missing_data(attribute):
    frequency = {}
    indices = []
    for i, data in enumerate(attribute):
        if data != "?":
            frequency[data] = frequency.get(data, 0) + 1
        else:
            indices.append(i)

    most_frequent_attribute = sorted(frequency, reverse=True, key=lambda x: frequency[x])[0]

    for index in indices:
        attribute[index] = most_frequent_attribute

    return attribute


def split_k_folds(rows, attribute_number, target_data_index, irrelevant_attr_index, k):
    target_data = []
    attribute_data = {}
    for attribute_index in range(attribute_number):
        attribute_data[attribute_index] = []

    rows = list(rows)
    random.shuffle(rows)

    for row in rows:
        attribute_index = 0
        for col_index, attribute in enumerate(row):
            if col_index != target_data_index and col_index not in irrelevant_attr_index:
                attribute_data[attribute_index].append(attribute)
                attribute_data[attribute_index] = attribute_data[attribute_index]
                attribute_index += 1
        target_data.append(row[target_data_index])

    fold_len = math.floor(len(target_data) / k)
    target_class_folds = {}
    attribute_data_folds = {}

    start_index = 0
    end_index = fold_len
    for i in range(k):
        target_class_folds[i] = target_data[start_index: end_index]
        attribute_data_folds[i] = {}
        for attr_index in attribute_data:
            attribute_data_folds[i][attr_index] = attribute_data[attr_index][start_index: end_index]
        start_index = end_index
        end_index += fold_len

    return target_class_folds, attribute_data_folds


def get_train_test_data(target_class_folds, attribute_data_folds, test_fold_index):
    train_class_data = []
    train_attribute_data = {}
    test_class_data = []
    test_attribute_data = {}

    attribute_temp = None
    for fold_index in target_class_folds:
        if fold_index == test_fold_index:
            test_class_data.extend(target_class_folds[fold_index])

            for attribute_index in attribute_data_folds[fold_index]:
                attribute_temp = test_attribute_data.get(attribute_index, None)
                if attribute_temp is None:
                    attribute_temp = []
                attribute_temp.extend(attribute_data_folds[fold_index][attribute_index])
                test_attribute_data[attribute_index] = attribute_temp
        else:
            train_class_data.extend(target_class_folds[fold_index])
            for attribute_index in attribute_data_folds[fold_index]:
                attribute_temp = train_attribute_data.get(attribute_index, None)
                if attribute_temp is None:
                    attribute_temp = []
                attribute_temp.extend(attribute_data_folds[fold_index][attribute_index])
                train_attribute_data[attribute_index] = attribute_temp

    return train_class_data, train_attribute_data, test_class_data, test_attribute_data

