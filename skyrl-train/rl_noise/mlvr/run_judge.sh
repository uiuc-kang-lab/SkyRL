#!/bin/bash
#SBATCH --account=daniel_kang
#SBATCH --partition=schmidt_sciences_cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1-00:00:00
#SBATCH --job-name=judge
#SBATCH --mem=8g

# Check that the GPU is available

batch_id=$1
echo "Processing batch ID: $batch_id"

export OPENAI_API_KEY=

uv run judge_generations.py $batch_id