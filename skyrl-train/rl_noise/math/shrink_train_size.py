import datasets


def load_train_set(parquet_path):
    dataset = datasets.load_dataset("parquet", data_files=parquet_path, split="train")
    return dataset


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="~/data/gsm8k/train.parquet")
    parser.add_argument("--proportion", type=float, default=0.1, help="Proportion of the dataset to keep")
    parser.add_argument("--output_path", type=str, default="~/data/gsm8k/")

    args = parser.parse_args()

    train_set = load_train_set(parquet_path=args.dataset_path)
    print(f"Train set size: {len(train_set)}")

    # reduce the size to %
    reduced_size = int(args.proportion * len(train_set))
    # shuffle the dataset
    train_set = train_set.shuffle(seed=42)
    reduced_train_set = train_set.select(range(reduced_size))
    print(f"Reduced train set size: {len(reduced_train_set)}")
    reduced_train_set.to_parquet(f"{args.output_path}/train_{args.proportion:.2f}.parquet")
    print(f"Reduced train set saved to {args.output_path}/train_{args.proportion:.2f}.parquet")
    # verify the saved file
    new_train_set = load_train_set(parquet_path=f"{args.output_path}/train_{args.proportion:.2f}.parquet")
    print(f"New train set size: {len(new_train_set)}")


