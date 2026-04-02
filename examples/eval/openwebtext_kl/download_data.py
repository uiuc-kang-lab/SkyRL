from datasets import load_dataset
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download the OpenWebText dataset.")
    parser.add_argument("--output_file", type=str, default="./data/openwebtext", help="File path to save the downloaded dataset.")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples to download from the dataset.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling the dataset.")
    args = parser.parse_args()

    # load the OpenWebText dataset
    dataset = load_dataset("openwebtext", split="train")
    
    # shuffle
    dataset = dataset.shuffle(seed=args.seed)
    
    # select a subset of the dataset
    dataset = dataset.select(range(args.num_samples))
    
    # save the dataset to the specified output directory
    dataset.to_parquet(args.output_file)
    print(f"Downloaded {args.num_samples} samples from the OpenWebText dataset and saved to {args.output_file}.")

