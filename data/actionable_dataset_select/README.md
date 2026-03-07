---
dataset_info:
  features:
  - name: rejected
    list:
    - name: content
      dtype: string
    - name: role
      dtype: string
  - name: chosen
    list:
    - name: content
      dtype: string
    - name: role
      dtype: string
  - name: chosen_score
    dtype: 'null'
  - name: rejected_score
    dtype: 'null'
  splits:
  - name: train
    num_bytes: 51381799.26790446
    num_examples: 27737
  download_size: 29850132
  dataset_size: 51381799.26790446
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---
