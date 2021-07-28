# This is the repo for paper: [Knowledge Enhanced Neural Fashion Trend Forecasting](https://arxiv.org/pdf/2005.03297.pdf)

## Requirements
1. OS: Ubuntu 16.04 or higher version
2. python3
3. python modules: yaml, pytorch, tqdm, numpy, pickle

## Code Structure

1. The entry script is: train.py
2. The config file is: config.yaml
3. utility_omnious.py: the script for omnious dataloader
4. model: the folder for model files

## How to prepare the dataset

## How to Run
1. Download the dataset to "dataset/omnious/" folder

2. Change the hyper-parameters in the configure file config.yaml.

3. Run: train.py

## Error analysis
### To evaluate the model per influencer group:
Run 
```python
python evaluate_per_group.py --device 1 --dataset omnious --weights /path/to/model.pt --save_filename_tsv file_to_save_results.tsv
```
### To evaluate the model per fashion element:
Run
```python
python evaluate_per_element.py --device 1 --dataset omnious --weights /path/to/model.pt --save_filename_tsv file_to_save_results.tsv
```

## How to run inference
Run 
```python
 python inference_trend.py --weights /path/to/model.pt --group_name  'location:All__segment:Nano__target_age:All' --fashion_element 'color:Black' --upload_date "2020-09-13 02:22:28" --save_infernece_trend True
```
If save_inference_trend is Ture, the plot will be saved in /inference_plots

### Acknowledgement
This project is supported by the National Research Foundation, Prime Minister's Office, Singapore under its IRC@Singapore Funding Initiative.

<img src="https://github.com/mysbupt/KERN/blob/master/next.png" width = "297" height = "100" alt="next" align=center />
