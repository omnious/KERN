import os
import json
from collections import defaultdict
from tqdm import tqdm
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dateutil.parser import parse
import itertools
import datetime
import time
import torch
import random
import ast
import argparse


def date2week(upload_date):
    # e.g. upload_date =  "2020-11-14 10:27:43"
    ymd = upload_date.split()[0]
    date_series = pd.Series([ymd])
    date_series = date_series.map(lambda x: parse(x))
    [week_no] = date_series.dt.isocalendar().week.tolist()
    last_week_2019 = 20191229  # "2019-12-29" # 52
    last_week_2020 = 20210103  # "2021-01-03" # 53
    [year, month, day] = ymd.split("-")
    ymd = int(year + month + day)
    # print("relative week", week_no)
    if ymd > last_week_2019 and ymd < last_week_2020:
        week_no += 52
    elif ymd > last_week_2020:
        week_no += 52 + 53
    week_no -= 44
    return week_no


def parse_data(opt):
    oms_dataset = os.path.join(opt.dataset_path, "influencer_json")
    nonempty_oms_dataset = os.path.join(opt.dataset_path, "nonempty_influencer_json")
    N = opt.num_data_points
    candidate_attributes = ['item', 'color', 'length', 'fit', 'shape', 'neckline', 'collar',
                            'look', 'gender', 'sleeve_length', 'sleeve_shape', 'design_detail', 'material', 'print']

    fashion_data = {}
    for file in tqdm(glob.glob(os.path.join(nonempty_oms_dataset, '*.json'))):
        with open(file) as json_file:
            data = json.load(json_file)
            posts = data["posts"]

            for post in posts:
                upload_date = post["upload_date"]
                week_no = date2week(upload_date)
                thisPostFashionElements = []
                for bbox in post["bbox"]:
                    for attribute in candidate_attributes:
                        if not bbox[attribute]:
                            continue
                        uniques_elements = [ast.literal_eval(el1) for el1 in set([str(el2) for el2 in bbox[attribute]])]
                        for element in uniques_elements:
                            if not element["name"]:
                                continue

                            fashion_element = ":".join([attribute, element["name"]])
                            thisPostFashionElements.append(fashion_element)

                for FE in thisPostFashionElements:
                    if FE not in fashion_data.keys():
                        fashion_data[FE] = [[x, 0] for x in range(N)]
                        fashion_data[FE][week_no][1] += 1
                    else:
                        fashion_data[FE][week_no][1] += 1
    return fashion_data


def apply_ranking(fashion_data, opt):
    Tmax = opt.num_data_points
    for t in range(Tmax + 1):
        Nt = defaultdict(int)
        for fashion in list(fashion_data.keys()):
            category_name = fashion.split(":")[0]
            if t < len(fashion_data[fashion]):
                Nt[category_name] += list(fashion_data[fashion][t])[1]

        for fashion in list(fashion_data.keys()):
            category_name = fashion.split(":")[0]
            if Nt[category_name] == 0:
                continue
            if t < len(fashion_data[fashion]):
                fashion_data[fashion][t][1] /= float(Nt[category_name])

    return fashion_data


def normalize(fashion_data, eps=0.01):
    norm_fashion_data = {}
    norm_fashion_stat = {}

    for fashion in list(fashion_data.keys()):
        week_num = fashion_data[fashion]
        timeline = [int(x[0]) for x in week_num]
        trend = [x[1] for x in week_num]

        max_v = max(trend)
        min_v = min(trend)
        normed_trend = [max((x - min_v) / (max_v - min_v + 1e-20), eps) for x in trend]
        res = []
        for time_s, trend_v in zip(timeline, normed_trend):
            res.append([time_s, trend_v])

        if fashion not in norm_fashion_data.keys():
            norm_fashion_data[fashion] = res
            norm_fashion_stat[fashion] = [min_v, max_v, eps]
        else:
            norm_fashion_data[fashion] = res
            norm_fashion_stat[fashion] = [min_v, max_v, eps]

    return norm_fashion_data, norm_fashion_stat

def main(opt):
    fashion_data = parse_data(opt)
    fashion_data = apply_ranking(fashion_data, opt)
    norm_fashion_data, norm_fashion_stat = normalize(fashion_data)

    data_filename = os.path.join("./dataset/omnious/", opt.save_data_file)
    data_norm_filename = os.path.join("./dataset/omnious/", opt.save_data_norm_file)
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return super(NpEncoder, self).default(obj)

    with open(data_filename, "w") as outfile:
        json.dump(norm_fashion_data, outfile, cls=NpEncoder)

    with open(data_norm_filename, "w") as outfile_norm:
        json.dump(norm_fashion_stat, outfile_norm, cls=NpEncoder)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='./Influencer_Export_autotagged/',
                        help="path to data with files")
    parser.add_argument("--num_data_points", type=int, default=88, help='number of data points')

    parser.add_argument('--save_data_file', default='new_data.json', help='name of created data .json file')
    parser.add_argument('--save_data_norm_file', default='new_data_norm.json',
                        help='name of created data norm .json file')

    opt = parser.parse_args()
    main(opt)


