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
    num_bytes: 102765451
    num_examples: 55475
  download_size: 59771028
  dataset_size: 102765451
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---
