import os
import json
from collections import defaultdict
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools
import datetime
import time
import random
import ast
import argparse
from utils.influ_group import InfluGroup
from utils.white_noise import WhiteNoise
from datetime import timedelta



def date2point(upload_date, start_date):
    '''
    Change date to data points by calculating how many weeks have passed since the start date
        - start date = 2019-11-01, end point = 2021-06-28
        - total : 88 data points
    '''

    upload_date, start_date = pd.to_datetime(upload_date), pd.to_datetime(start_date)
   
    upload_date = upload_date - timedelta(upload_date.weekday())
    start_date = start_date - timedelta(start_date.weekday())
    
    data_point = upload_date.replace(hour=0, minute=0, second=0) - start_date.replace(hour=0, minute=0, second=0)
    data_point = int(data_point / np.timedelta64(1, 'W'))
    
    return data_point


# Covert data point to week number of the year
def point2week(w, start_date):
    upload_date = pd.to_datetime(start_date) + timedelta(weeks=w)
    return upload_date.week - 1


# Remove spare time series
def remove_sparse_seq(data, rate=0.5):
    removed = 0
    total = 0
    # remove time series with sparsity >= rate
    for group in data.keys():
        for FE, res in list(data[group].items()):
            per = len([x[1] for x in res if x[1] == 0]) / len(res)
            if per >= rate:
                removed += 1
                del data[group][FE]
            total += 1
    print("removed : " + str(removed) + " total : " + str(total))
    return data


# Using permutations of group attr and fashion element, create pairs of group attr and fashion elements
def create_IG_FE_Pairs(opt, post, influGroup):
    
    postFE = []
    for bbox in post["bbox"]:
        for attribute in opt.candidate_attributes:
            # if bbox contains candidate attributes
            if bbox[attribute]:
                # if there are multiple same Fashion Element, select only one
                uniques_elements = [ast.literal_eval(el1) for el1 in set([str(el2) for el2 in bbox[attribute]])]
                for element in uniques_elements:
                    if element["name"]:
                        fe = ":".join([attribute, element["name"]])
                        postFE.append(fe)
        influ_FE_Pairs = [influGroup, postFE]
        influ_FE_Pairs = list(itertools.product(*influ_FE_Pairs))
        
        return influ_FE_Pairs


def parse_data(opt):
    
    # Json file location
    dataset_path = os.path.join(opt.dataset_path, opt.json_dir)
    
    # set which group attribute to use
    setting = {'location' : opt.use_loc, 'segment': opt.use_seg, 'target_age' : opt.use_age}
    use_all = opt.use_all
    drop_attr = opt.drop_attr
    T = date2point(opt.end_date, opt.start_date) + 1
    replace_dict = json.load(open(opt.replace_dict_path))

    all_data = {}
    
    for file in glob.glob(os.path.join(dataset_path, '*.json')):
        
        with open(file) as json_file:
            data = json.load(json_file)

            # create permutations of group attributes
            currInfluGroups = InfluGroup().create_permu(data, setting, use_all, replace_dict, drop_attr)
            
            # if some of group attributes are missing in json file, skip that
            if not currInfluGroups:
                continue
            
            posts = data["posts"]
            
            for post in posts:
                
                upload_date = post["upload_date"]
                # skip outdated post (post before the start date)
                if pd.to_datetime(upload_date) < pd.to_datetime(opt.start_date):
                    continue
                
                # create pairs of group attr and fashion elements
                IG_FE_Pairs = create_IG_FE_Pairs(opt, post, currInfluGroups)
                
                # convert upload date to data point
                data_point = date2point(upload_date, opt.start_date)
            
                for (IG, FE) in IG_FE_Pairs:
                    
                    # all_data format : all_data[IG] = {FE: [[week_no, 1]]}
                    if IG not in all_data.keys():
                        all_data[IG] = {FE: [[x, 0] for x in range(T)]}
                    if FE not in all_data[IG].keys():
                        all_data[IG][FE] = [[x, 0] for x in range(T)]
                    all_data[IG][FE][data_point][1] += 1
    
    return all_data


# Convert all of the datapoints in timeseries into week number
def format_weeks(opt, data):
    
    for influencer in data.keys():
        for fashion in data[influencer].keys():
            for i in range(len(data[influencer][fashion])):
                data[influencer][fashion][i][0] = point2week(data[influencer][fashion][i][0], opt.start_date)

    return data


def ranking(opt, data):
    '''
    calculate the popluarity of fashion element of each group for each timestep

        * category : score popularity among fashion category in current influencer group

        * num_group : score popluarity among all fashion elements in current influencer group
    '''
    assert opt.ranking in ['category', 'num_group']
    
    # calculate max datapoint in timeseries
    Tmax = date2point(opt.end_date, opt.start_date) + 1

    for influencer in list(data.keys()):
        for t in range(Tmax):
            Nt = defaultdict(int)
            for fashion in list(data[influencer].keys()):
                if opt.ranking == 'category':
                    denominator = fashion.split(":")[0]
                else:
                    denominator = influencer 
                Nt[denominator] += list(data[influencer][fashion][t])[1]
            
            for fashion in data[influencer].keys():
                if opt.ranking == 'category':
                    denominator = fashion.split(":")[0]
                else:
                    denominator = influencer
                if not Nt[denominator] == 0:
                    data[influencer][fashion][t][1] /= float(Nt[denominator])
    
    return data


# Perform min-max normalization to timeseries
def normalize(data, eps=0.01):
    norm_all_data = {}
    norm_data_stat = {}
    count = 0

    for influencer in data.keys():
        for fe, res in data[influencer].items():
            data_points = [x[0] for x in res]
            trend = [x[1] for x in res]
            if not (trend or data_points):
                count += 1
                continue
        
            max_v, min_v = max(trend), min(trend)
            normed_trend = [max((x-min_v)/(max_v-min_v + 1e-20), eps) for x in trend]
            
            res_new = []
            for time_s, trend_v in zip(data_points, normed_trend):
                res_new.append([time_s, trend_v])
        
            if influencer not in norm_all_data.keys():
                norm_all_data[influencer] = {fe: res_new}
                norm_data_stat[influencer] = {fe: [min_v, max_v, eps]}
            else:
                norm_all_data[influencer][fe] = res_new
                norm_data_stat[influencer][fe] = [min_v, max_v, eps]
    print('count: '+str(count))
    return norm_all_data, norm_data_stat


def main(opt):
    
    # parse data from json file
    data = parse_data(opt)
    
    # remove timeseries where percentage of zero values is more than sparsity
    if opt.sparsity > 0:
        data = remove_sparse_seq(data)
    
    # convert datapoints in timeseries into week num
    data = format_weeks(opt, data)
    

    # calculate popularity score, popularity among the group or among fashion element category in group
    data = ranking(opt, data)
    
    # apply min-max normalization to time-series
    norm_data, norm_stat = normalize(data)

    # filter out white noise
    if opt.filter_wn:
        norm_data, norm_stat, filtered = WhiteNoise.filter_WN(norm_data, norm_stat)

    
    # saving the files
    if not os.path.exists(opt.save_file_dir):
        os.makedirs(opt.save_file_dir)
    
    data_filename = os.path.join(opt.save_file_dir, opt.save_data_file)
    data_norm_filename = os.path.join(opt.save_file_dir, opt.save_data_norm_file)

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
        json.dump(norm_data, outfile, cls=NpEncoder)

    with open(data_norm_filename, "w") as outfile_norm:
        json.dump(norm_stat, outfile_norm, cls=NpEncoder)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--dataset_path", type=str, default='./Influencer_Export_autotagged/',
                        help="path to data with files")
    
    parser.add_argument("--json_dir", type=str, default='influencer_json', help='dir contains all of json files')

    parser.add_argument("--replace_dict_path", type=str, default= './dataset/replace.json', help="location of dict about fined group attribute to be replace")

    parser.add_argument("--start_date", type=str, default="2019-11-01 18:33:22", help='start date of the timeseries')

    parser.add_argument("--end_date", type=str, default="2021-06-28 07:07:42", help='end date of the timeseries')

    parser.add_argument("--sparsity", type=int, default=0.5,
                        help="if sparsity > 0, remove sequences with sparsity*100 sequnces ")
    
    parser.add_argument('--save_file_dir', type=str, default="./dataset/omnious/", help='path of directory where preprocessing json file saved')
    
    parser.add_argument('--save_data_file', default='omnious_trend.json',
                        help='name of created data norm .json file')

    parser.add_argument('--save_data_norm_file', default='omnious_trend_norm.json',
                        help='name of created data norm .json file')
    
    parser.add_argument('--candidate_attributes', type=list, 
                        default= ['item', 'color', 'length', 'fit', 'shape', 'neckline', 'collar', 'look', 
                        'gender', 'sleeve_length', 'sleeve_shape', 'design_detail', 'material', 'print'],
                        help='list of fashion elements to be used to create dataset')
    
    parser.add_argument("--use_loc", type=bool, default=True, help="whether use location as user attribute")

    parser.add_argument("--use_seg", type=bool, default=True, help="whether use segment as user attribute")

    parser.add_argument("--use_age", type=bool, default=True, help="whether use location as user attribute")

    parser.add_argument("--use_all", type=bool, default=False, help="whether use merged group or not")

    parser.add_argument("--drop_attr", type=list, default=[], help="drop group attributes in list and create new influencer group")

    parser.add_argument("--filter_wn", type=bool, default=True, help="whether filter out white noise or not")

    parser.add_argument("--ranking", type=str, default='category', help="choose whether to caculate trend score based on \
                        number of fashion elements in fashion element category or user group")

    opt = parser.parse_args()
    main(opt)