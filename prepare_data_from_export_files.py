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


def format_week(w):
    if w <= 8:
        return w + 44
    elif w <= 61:
        return w - 8
    else:
        return w - 61


def parse_data(opt):
    oms_dataset = os.path.join(opt.dataset_path, "influencer_json")
    nonempty_oms_dataset = os.path.join(opt.dataset_path, "nonempty_influencer_json")
    N = opt.num_data_points
    MERGE = opt.merge_groups
    candidate_attributes = ['item', 'color', 'length', 'fit', 'shape', 'neckline', 'collar',
                            'look', 'gender', 'sleeve_length', 'sleeve_shape', 'design_detail', 'material', 'print']
    all_data = {}

    for file in tqdm(glob.glob(os.path.join(nonempty_oms_dataset, '*.json'))):
        with open(file) as json_file:
            data = json.load(json_file)
            posts = data["posts"]

            # Create influencer groups
            locations = ["location:" + x["name"] for x in data["location"]]
            # merge locations
            if MERGE:
                locations.append("location:All")
            segments = ["segment:" + x["name"].split()[0].strip() for x in data["segment"]]
            # merge segments
            if MERGE:
                merged_segments = []
                if "segment:Nano" in segments or "segment:Micro" in segments:
                    merged_segments.append("segment:Nano_AND_Micro")
                merged_segments.append("segment:All")
                segments.extend(merged_segments)

            target_ages = []

            for age in data["target_age"]:
                target_ages.append("target_age:" + age["name"].split("''")[0].strip())

            # merge target_ages
            if MERGE:
                merged_target_ages = []
                if "target_age:10's" in target_ages or "target_age:20's" in target_ages:
                    merged_target_ages.append("target_age:10's_AND_20's")
                if "target_age:30's" in target_ages or "target_age:20's" in target_ages:
                    merged_target_ages.append("target_age:20's_AND_30's")
                if "target_age:30's" in target_ages or "target_age:40's" in target_ages:
                    merged_target_ages.append("target_age:30's_AND_40's")
                merged_target_ages.append("target_age:All")
                target_ages.extend(merged_target_ages)

            forPermutation = [locations, segments, target_ages]
            allPermutations = list(itertools.product(*forPermutation))
            currInfluencerGroups = []
            for p in allPermutations:
                influencer_group_name = "__".join(p)
                currInfluencerGroups.append(influencer_group_name)

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
                influencerFashionElementPairs = [currInfluencerGroups, thisPostFashionElements]
                influencerFashionElementPairs = list(itertools.product(*influencerFashionElementPairs))

                for (IG, FE) in influencerFashionElementPairs:
                    if IG not in all_data.keys():
                        # all_data[IG] = {FE: [[week_no, 1]]}
                        all_data[IG] = {FE: [[x, 0] for x in range(N)]}
                        all_data[IG][FE][week_no][1] += 1
                    elif FE not in all_data[IG].keys():
                        # all_data[IG][FE] = [[week_no, 1]]
                        all_data[IG][FE] = [[x, 0] for x in range(N)]
                        all_data[IG][FE][week_no][1] += 1
                    else:
                        all_data[IG][FE][week_no][1] += 1
    return all_data


def remove_sparse(opt, all_data):
    sparsity = opt.sparsity

    removed = 0
    total = 0
    for influencer in tqdm(list(all_data.keys())):
        for fashion in list(all_data[influencer].keys()):
            zeros = 0
            for i in range(len(all_data[influencer][fashion])):
                if all_data[influencer][fashion][i][1] == 0:
                    zeros += 1

            zeros = 100 * zeros / opt.num_data_points
            if zeros >= 100 * sparsity:
                del all_data[influencer][fashion]
                removed += 1
            total += 1

    print("removed: ", 100 * removed / total)
    return all_data


def format_weeks(all_data):
    for influencer in tqdm(list(all_data.keys())):
        for fashion in list(all_data[influencer].keys()):
            for i in range(len(all_data[influencer][fashion])):
                all_data[influencer][fashion][i][0] = format_week(all_data[influencer][fashion][i][0])

    return all_data


def apply_ranking(opt, all_data):
    for influencer in tqdm(list(all_data.keys())):
        for t in range(opt.num_data_points + 1):
            Nt = defaultdict(int)
            for fashion in list(all_data[influencer].keys()):
                category_name = fashion.split(":")[0]
                if t < len(all_data[influencer][fashion]):
                    Nt[category_name] += list(all_data[influencer][fashion][t])[1]

            for fashion in list(all_data[influencer].keys()):
                category_name = fashion.split(":")[0]
                if Nt[category_name] == 0:
                    continue
                if t < len(all_data[influencer][fashion]):
                    all_data[influencer][fashion][t][1] /= float(Nt[category_name])

    return all_data


def normalize(all_data, eps=0.01):
    norm_all_data = {}
    norm_data_stat = {}

    for influencer in tqdm(list(all_data.keys())):
        for fashion in list(all_data[influencer].keys()):
            week_num = all_data[influencer][fashion]
            timeline = [int(x[0]) for x in week_num]
            trend = [x[1] for x in week_num]
            if len(trend) == 0 or len(timeline) == 0:
                print("empty 1: ", influencer, fashion, len(trend), len(timeline))
                continue

            max_v = max(trend)
            min_v = min(trend)
            normed_trend = [max((x - min_v) / (max_v - min_v + 1e-20), eps) for x in trend]
            res = []
            for time_s, trend_v in zip(timeline, normed_trend):
                res.append([time_s, trend_v])

            if len(res) == 0:
                print("empty 2: ", influencer, fashion, len(res))

            if influencer not in norm_all_data.keys():
                norm_all_data[influencer] = {fashion: res}
                norm_data_stat[influencer] = {fashion: [min_v, max_v, eps]}
            else:
                norm_all_data[influencer][fashion] = res
                norm_data_stat[influencer][fashion] = [min_v, max_v, eps]

    return norm_all_data, norm_data_stat


def main(opt):
    all_data = parse_data(opt)
    if opt.sparsity > 0:
        all_data = remove_sparse(opt, all_data)
    all_data = format_weeks(all_data)
    all_data = apply_ranking(opt, all_data)
    norm_all_data, norm_data_stat = normalize(all_data)
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
        json.dump(norm_all_data, outfile, cls=NpEncoder)

    with open(data_norm_filename, "w") as outfile_norm:
        json.dump(norm_data_stat, outfile_norm, cls=NpEncoder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='./Influencer_Export_autotagged/',
                        help="path to data with files")
    parser.add_argument('--merge_groups', type=bool, default=True, help='merge some influencer groups if True')
    parser.add_argument("--num_data_points", type=int, default=88, help='number of data points')
    parser.add_argument("--sparsity", type=int, default=0.5,
                        help="if sparsity > 0, remove sequences with sparsity*100 sequnces ")
    parser.add_argument('--save_data_file', default='new_data.json', help='name of created data .json file')
    parser.add_argument('--save_data_norm_file', default='new_data_norm.json',
                        help='name of created data norm .json file')

    opt = parser.parse_args()
    main(opt)



