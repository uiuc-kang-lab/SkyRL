# DeepCoder + LCB Run 

## Download Dataset 
```
pip install gdown

python examples/livecodebench/lcb_download.py --local_dir ~/data/lcb/download

python examples/livecodebench/lcb_dataset.py --dataset_dir ~/data/lcb/download --local_dir ~/data/lcb/
```

## Note
* Read from the json file instead of parquet 
* Need to truncate the json file otherwise it is too large with normal `datasets.load_dataset()`, need to use streaming or load with PyArrow directly 

## Prepare Noise Data
1. sample wrong test using `sample_wrong_tests.sh`
```bash
bash rl_noise/code/sample_wrong_tests.sh --model=Qwen/Qwen3-8B --run_name=wrong-answers_qwen3-8b 
```

2. check whether 1 returns enough wrong tests, if not, repeat it

3. construct a dataset with wrong tests using `collect_gentests.py`, with line #9, #79, and #84 modified
```bash
uv run rl_noise/code/collect_gentests.py
```