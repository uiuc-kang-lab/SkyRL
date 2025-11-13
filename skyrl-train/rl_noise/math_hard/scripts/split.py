import argparse
import json


parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str, required=True)
parser.add_argument("--n_split", type=int, default=4)

args = parser.parse_args()

with open(args.input_file) as f:
    data = json.load(f)

n_data = len(data)
split_size = n_data // args.n_split
for i in range(args.n_split):
    split_data = data[i * split_size: (i + 1) * split_size] if i < args.n_split - 1 else data[i * split_size:]
    with open(f"{args.input_file.rstrip('.json')}_part_{i}.json", "w") as f:
        json.dump(split_data, f, indent=4)
