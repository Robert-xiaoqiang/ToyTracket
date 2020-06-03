# preprocess
```bash
# configure "mgtv_root_path" and "preprocess_output_path" firstly
python package/train/dataset/mgtv/split.py
# configure "train_preprocessed_path" firstly
python package/train/dataset/crop_image.py
```

# train
```bash
# configure "crop_base_path" and "save_path"(ckpt path) firstly
python train.py
```

# test on validating set
```bash
# configure "data_root_path" and "result_output_path"
python test.py
```