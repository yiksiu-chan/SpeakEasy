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
    num_bytes: 39009085.17047319
    num_examples: 27738
  download_size: 23238403
  dataset_size: 39009085.17047319
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---
