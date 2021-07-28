import yaml
from tqdm import tqdm
import numpy as np
import timeit
import os
import glob
import pandas as pd
from dateutil.parser import parse
import json
import argparse

import torch
import matplotlib.pyplot as plt
import json
from collections import defaultdict

from model.KERN import KERN
from model.KERN_oms import KERN_oms


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


def denorm(seq, norms):
    [min_v, max_v, eps] = norms
    temp_seq = [x[1] for x in seq]
    temp_axis = [x[0] for x in seq]
    denorm_pred = [x * (max_v - min_v) + min_v for x in temp_seq]
    denorm_res = []
    for i in range(len(temp_axis)):
        denorm_res.append([temp_axis[i], denorm_pred[i]])

    return denorm_res


def inference(model, conf, group_name, fashion_element, upload_date, denorm_flag=True):
    logdir = os.path.join('./log/' + conf["data_path"].split('/')[-1].split('.')[0])
    location_filename = os.path.join(logdir, "location_id_map.json")
    segment_filename = os.path.join(logdir, "segment_id_map.json")
    target_age_filename = os.path.join(logdir, "target_age_id_map.json")
    ele_filename = os.path.join(logdir, "ele_id_map.json")
    grp_filename = os.path.join(logdir, "grp_id_map.json")
    device = conf["device"]

    with open(location_filename) as location_file:
        location_id_map = json.load(location_file)

    with open(segment_filename) as segment_file:
        segment_id_map = json.load(segment_file)

    with open(target_age_filename) as target_age_file:
        target_age_id_map = json.load(target_age_file)

    with open(ele_filename) as ele_file:
        ele_id_map = json.load(ele_file)

    with open(grp_filename) as grp_file:
        grp_id_map = json.load(grp_file)

    with open(conf["data_path"]) as trends_json:
        trends_data = json.load(trends_json)

    with open(conf["data_norm_path"]) as norm_json:
        norm_data = json.load(norm_json)

    time_point = date2week(upload_date)
    trends = np.array(trends_data[group_name][fashion_element])
    input_len = conf["input_len"]
    output_len = conf["output_len"]
    comps = group_name.split("__")
    for each in comps:
        if "location:" in each:
            location_id = torch.LongTensor([np.array([location_id_map[each]])]).to(device)
        if "segment:" in each:
            segment_id = torch.LongTensor([np.array([segment_id_map[each]])]).to(device)
        if "target_age:" in each:
            target_age_id = torch.LongTensor([np.array([target_age_id_map[each]])]).to(device)

    ele_id = torch.LongTensor([np.array([ele_id_map[fashion_element]])]).to(device)
    grp_id = torch.LongTensor([np.array([grp_id_map[group_name]])]).to(device)
    input_seq = torch.Tensor(trends[time_point:time_point + input_len]).to(device)
    output_seq = torch.Tensor(trends[time_point - output_len: time_point]).to(device)

    input_seq = torch.unsqueeze(input_seq, 0)
    output_seq = torch.unsqueeze(output_seq, 0)

    self_cont = [input_seq, output_seq, grp_id, ele_id, [], location_id, segment_id, target_age_id]

    model.eval()
    [self_enc_loss, self_dec_loss, self_pred, self_enc_hidden, self_enc_output, self_dec_output] = model.predict(
        self_cont)

    trends = trends.tolist()
    pred = self_pred.tolist()[0]
    if denorm_flag:
        norms = norm_data[group_name][fashion_element]
        trends = denorm(trends, norms)
        pred = [[0, x] for x in pred]
        pred = denorm(pred, norms)
        pred = [x[1] for x in pred]
    return trends, time_point, pred


def visualize_inference(group_name, fashion_element, trends, upload_date, pred, save=False):
    time_point = date2week(upload_date)
    true_values = [x[1] for x in trends]
    true_x_axis = [i for i in range(len(trends))]

    pred_values = pred
    pred_x_axis = [time_point + x for x in range(4)]

    plt.figure(figsize=(20, 5))
    plt.plot(true_x_axis, true_values, 'ro', linestyle='dashed', color="blue")
    plt.plot(pred_x_axis, pred_values, 'ro', linestyle='dashed', color="red")

    title = group_name + "--" + fashion_element
    plt.suptitle(title)
    if save:
        image_filename = "inference_plots/" + title + "_" + str(upload_date.split()[0])
        plt.savefig(image_filename)
        print("Saved to ", image_filename)
    plt.show()


def inference_and_visualize(model, conf, group_name, fashion_element, upload_date, save=False):
    trends, time_point, pred = inference(model, conf, group_name, fashion_element, upload_date)
    visualize_inference(group_name, fashion_element, trends, upload_date, pred, save)


def main(opt):
    conf = yaml.load(open("./config.yaml"))
    dataset_name = opt.dataset
    conf = conf[dataset_name]
    conf["dataset"] = dataset_name
    if len(opt.data_path) > 0 and len(opt.data_norm_path) > 0:
        conf['data_path'] = opt.data_path
        conf['data_norm_path'] = opt.data_norm_path

    if dataset_name == "fit":
        from utility import TrendData
    elif dataset_name == "omnious":
        from utility_omnious import TrendData

    # Dataset
    dataset = TrendData(conf)
    conf["grp_num"] = len(dataset.grp_id_map)
    conf["ele_num"] = len(dataset.ele_id_map)
    conf["time_num"] = dataset.time_num
    if dataset_name == "fit":
        conf["city_num"] = len(dataset.city_id_map)
        conf["gender_num"] = len(dataset.gender_id_map)
        conf["age_num"] = len(dataset.age_id_map)
    elif conf["dataset"] == "omnious":
        conf["location_num"] = len(dataset.location_id_map)
        conf["segment_num"] = len(dataset.segment_id_map)
        conf["target_age_num"] = len(dataset.target_age_id_map)

    conf["device"] = opt.device
    device = conf["device"]

    # Model
    if dataset_name == "fit":
        model = KERN(conf, dataset.adj)
        weights = opt.weights
    elif conf['dataset'] == "omnious":
        model = KERN_oms(conf)
        weights = opt.weights
    model.to(device)

    ckpt = torch.load(weights, map_location=device)
    state_dict = ckpt['model']
    model.load_state_dict(state_dict, strict=False)

    inference_and_visualize(model, conf, opt.group_name, opt.fashion_element, opt.upload_date,
                            save=opt.save_infernece_trend)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='omnious', help='dataset name: omnious')
    parser.add_argument('--device', default='cuda:0', help='cuda device, i.e. cuda:0')
    parser.add_argument('--weights', type=str, default='', help='model weights path')
    parser.add_argument('--group_name', type=str, default="location:Local__segment:Micro__target_age:40's",
                        help='influencer group name')
    parser.add_argument('--fashion_element', type=str, default="item: Coat", help='fashion element')
    parser.add_argument('--upload_date', type=str, default="2020-08-13 02:22:28", help="predict trend from upload_date")
    parser.add_argument('--data_path', type=str, default='', help='path to data')
    parser.add_argument('--data_norm_path', type=str, default='', help='path to norm data')
    parser.add_argument('--save_infernece_trend', type=bool, default=False, help="if True save trend to inference_plot")

    opt = parser.parse_args()
    main(opt)
