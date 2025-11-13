#!/bin/bash
#SBATCH --account=daniel_kang
#SBATCH --partition=schmidt_sciences_cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1-00:00:00
#SBATCH --job-name=diff
#SBATCH --mem=8g

export OPENAI_API_KEY=

uv run diff_answers_gpt.py --mode $1 --part_id $2