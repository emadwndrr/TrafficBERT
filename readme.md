# Code Structure

## `pcap` directory
Contains the code that is used to parse the pcap files and extract the flows (`flow.py`), compute and plot the dataset and flow statistics (`plot.py` and `stats.py`), and create the features from flows (`featuers.py`)

## `generated_dataset`
produced by `create_datasets.py`. contains the generated normalized dataset.

## `pcap_datasets`
Contains the raw pcap files from the `USTC-TFC2016` dataset.

## `results`
Contains the task agnostic plots, and task-specific (`20class` and `10class`) trained models an plots.

## `choose_n.py`
Contains code that plots the flow length cdf and flow length distribution plots used to derive `N=20`.

## `compute_raw_stats.py`
Contains code that computes the dataste packet count plots i.e. ipv6 vs ipv4, tcp+udp pe class

## `create_dataset.py`
Contains code to extract the flows and compute the features and normalize the features and save them to the `generated_datasets` directory

## `plot_feature_overlap.py`
Contains code that plots t-SNE for the aggregated features (mean) for `N=20`. also code that plots feature mean per position in the input sequence, and per class feature distribution plots.
**important** set the `TASK` variable to either `10class` or `20class`


## `train.py` 
Contains code that defines the model and train it and saves the result (model, confusion matrix, training plots) to the `results/<task>`  directory/
**important** set the `TASK` variable to either `10class` or `20class`

## `evaluate.py`
Contains code that loads the model form `results/<task>` and evaluates it and re-produces the confusion matrix plot.
**important** set the `TASK` variable to either `10class` or `20class`

# Where to get the USTC-TFC2016 data

get the data from [https://www.kaggle.com/datasets/randasrour/ustctfc2016/](https://www.kaggle.com/datasets/randasrour/ustctfc2016/) and extract `archive.zip` to `pcap_datasets`

# Generate Datasets
To generated the dataset run `python create_datasets.py` 


# Train and Evaluate

to train run `python train.py` to evaluate run `python evaluate.py`

 
