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
    num_bytes: 78016764
    num_examples: 55475
  download_size: 46555220
  dataset_size: 78016764
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---
