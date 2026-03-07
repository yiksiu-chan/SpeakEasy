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
    num_bytes: 39007678.82952681
    num_examples: 27737
  download_size: 23329818
  dataset_size: 39007678.82952681
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---
