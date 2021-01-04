#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""convert adult dataset

workclass::
    employed: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov
    not employed: Without-pay, Never-worked
education::
    university: Bachelors,Masters,Doctorate
    not university: Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, 1st-4th, 10th, 5th-6th, Preschool
marital-status::
    married: Married-civ-spouse
    single: Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse
sex::
    male/female
native-country::
    United-States: United-States
    others: - United States
income:
    <=50K/>50K
"""
import os
import re


def discretize_new():
    adult_dir = os.getcwd()
    adult_raw_path = os.path.join(adult_dir, "adult.data.txt")

    with open(adult_raw_path, "r") as fp:
        raw_data = fp.read()

    indices = [1, 3, 6, 14]
    fimi_data = []
    rows = raw_data.split("\n")

    next_alphabet = 1
    value_to_alphabet = dict()
    for row_num, row in enumerate(rows):
        if not row or "?" in row:
            continue

        fimi_row = []
        vals = map(str.strip, row.split(","))
        for pos, idx in enumerate(indices):
            val = vals[idx]
            if value_to_alphabet.has_key(val):
                fimi_row.append(value_to_alphabet[val])
            else:
                fimi_row.append(next_alphabet)
                value_to_alphabet[val] = next_alphabet
                next_alphabet += 1
        fimi_data.append(" ".join(str(v) for v in fimi_row))

    adult_dat_path = os.path.join(adult_dir, "adult.dat")
    with open(adult_dat_path, "w") as fp:
        fp.write("\n".join(row for row in fimi_data))


def discretize():
    substitutes = {
        "Private": "private",
        "Self-emp-not-inc": "self_employed",
        "Self-emp-inc": "self_employed",
        "Federal-gov": "public_servant",
        "Local-gov": "public_servant",
        "State-gov": "public_servant",
        "Without-pay": "unemployed",
        "Never-worked": "unemployed",

        "Adm-clerical": "admin",
        "Armed-Forces": "armed_forces",
        "Craft-repair": "blue_collar",
        "Handlers-cleaners": "blue_collar",
        "Machine-op-inspct": "blue_collar",
        "Farming-fishing": "blue_collar",
        "Transport-moving": "blue_collar",
        "Other-service": "service",
        "Priv-house-serv": "service",
        "Sales": "sales",
        "Exec-managerial": "white_collar",
        "Prof-specialty": "professional",
        "Tech-support": "other_occupations",
        "Protective-serv": "other_occupations",

        "10th": "dropout",
        "11th": "dropout",
        "12th": "dropout",
        "1st-4th": "dropout",
        "5th-6th": "dropout",
        "7th-8th": "dropout",
        "9th": "dropout",
        "Assoc-acdm": "associates",
        "Assoc-voc": "associates",
        "Bachelors": "bachelors",
        "Doctorate": "doctorate",
        "Some-college": "hs_graduate",
        "HS-grad": "hs_graduate",
        "Masters": "masters",
        "Preschool": "dropout",
        "Prof-school": "prof_school",

        "Married-civ-spouse": "married",
        "Married-AF-spouse": "married",
        "Married-spouse-absent": "single",
        "Divorced": "single",
        "Never-married": "single",
        "Separated": "single",
        "Widowed": "single",

        "United-States": "US_native",
        "Cambodia": "non_native", "England": "non_native", "Puerto-Rico": "non_native", "Canada": "non_native", "Germany": "non_native", "Outlying-US(Guam-USVI-etc)": "non_native", "India": "non_native", "Japan": "non_native", "Greece": "non_native", "South": "non_native", "China": "non_native", "Cuba": "non_native", "Iran": "non_native", "Honduras": "non_native", "Philippines": "non_native", "Italy": "non_native", "Poland": "non_native", "Jamaica": "non_native", "Vietnam": "non_native", "Mexico": "non_native", "Portugal": "non_native", "Ireland": "non_native", "France": "non_native", "Dominican-Republic": "non_native", "Laos": "non_native", "Ecuador": "non_native", "Taiwan": "non_native", "Haiti": "non_native", "Columbia": "non_native", "Hungary": "non_native", "Guatemala": "non_native", "Nicaragua": "non_native", "Scotland": "non_native", "Thailand": "non_native", "Yugoslavia": "non_native", "El-Salvador": "non_native", "Trinadad&Tobago": "non_native", "Peru": "non_native", "Hong": "non_native", "Holand-Netherlands": "non_native"
    }

    indices = [1, 3, 6, 14]

    categorical_values = [
        ("private", "self_employed", "public_servant", "unemployed"),
        ("dropout", "associates", "bachelors", "doctorate",
         "hs_graduate", "masters", "prof_school"),
        # ("married", "single"), # index=5
        ("admin", "armed_forces", "blue_collar", "white_collar",
         "service", "sales", "professional", "other_occupations"),
        # ("US_native", "non_native"), # index=13
        ("<=50K", ">50K")
    ]
    alphabet = []
    previous = 0
    for values in categorical_values:
        alphabet.append(range(previous, previous + len(values)))
        previous += len(values)

    adult_dir = os.getcwd()
    adult_raw_path = os.path.join(adult_dir, "adult.data.txt")

    with open(adult_raw_path, "r") as fp:
        raw_data = fp.read()

    for word, substitute in substitutes.iteritems():
        raw_data = re.sub(r"%s" % word, substitute, raw_data)

    fimi_data = []
    rows = raw_data.split("\n")

    for row_num, row in enumerate(rows):
        if not row:
            continue

        fimi_row = []
        vals = map(str.strip, row.split(","))

        for pos, dat_idx in enumerate(indices):
            val = vals[dat_idx]
            print val,
            try:
                val_idx = categorical_values[pos].index(val)
                fimi_row.append(alphabet[pos][val_idx])
            except ValueError:
                pass
        print

        if len(fimi_row) != len(categorical_values):
            continue

        fimi_data.append(" ".join(str(v) for v in fimi_row))

    # write fimi dat file
    adult_dat_path = os.path.join(adult_dir, "adult.dat")
    with open(adult_dat_path, "w") as fp:
        fp.write("\n".join(row for row in fimi_data))

    labels = [val for values in categorical_values for val in values]
    adult_labels_path = os.path.join(adult_dir, "adult.labels")
    with open(adult_labels_path, "w") as fp:
        fp.write("\n".join(label for label in labels))


if __name__ == "__main__":
    discretize_new()
