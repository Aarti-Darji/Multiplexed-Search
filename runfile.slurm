#!/bin/bash

#SBATCH --job-name=codex_gray_colon_128_256
#SBATCH --partition=normal
#SBATCH --account=conference
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hajighasemiamir@gmail.com
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=120:00:00
#SBATCH --error=/home/axh5735/projects/MPIR/MPIR-2.0/logs/codex_logs/codex_gray_colon_128_256/slurm_logs/codex_gray_colon_128_256.err
#SBATCH --output=/home/axh5735/projects/MPIR/MPIR-2.0/logs/codex_logs/codex_gray_colon_128_256/slurm_logs/codex_gray_colon_128_256.out




function fail {
    echo "FAIL: $@" >&2
    exit 1  # signal failure
}

source /home/axh5735/miniconda3/etc/profile.d/conda.sh || fail "conda load fail"
conda activate IC || fail "conda activate fail"

cd '/home/axh5735/projects/MPIR/MPIR-2.0/codex_Model'


python /home/axh5735/projects/MPIR/MPIR-2.0/codex_Model/runfile.py \
    --transformations_write_dir /home/axh5735/projects/MPIR/MPIR-2.0/logs/codex_logs/codex_gray_colon_128_256/transformation \
    --inv_transformations_read_dir /home/axh5735/projects/MPIR/MPIR-2.0/logs/codex_logs/codex_gray_colon_128_256/transformation \
    --transformations_read_dir /home/axh5735/projects/MPIR/MPIR-2.0/logs/codex_logs/codex_gray_colon_128_256/transformation \
    --patch_size 64 \
    --num_patches_per_image 70 \
    --whitespace_threshold 0.82 \
    --selected_channel 0 \
    --test_ratio 0.2 \
    --val_ratio 0.2 \
    --everything_seed 1 \
    --split_seed 2 \
    --shuffling_seed 3 \
    --patching_seed 4 \
    --logging_dir /home/axh5735/projects/MPIR/MPIR-2.0/logs/codex_logs/codex_gray_colon_128_256/tb_logs \
    --logging_name codex_gray_colon_128_256 \
    --batch_size 128 \
    --num_dataloader_workers 8 \
    --input_height 64 \
    --latent_dim 256 \
    --lr 0.0001 \
    --gradient_clip_val 0.5 \
    --normalize_transform \
    --num_nodes 1 \
    --accelerator gpu \
    --devices 1 \
    --max_epochs 200 \
    --auto_lr_find False \
    --auto_scale_batch_size False \
    --benchmark True \
    --fast_dev_run False \
    --strategy ddp \
    --coords_write_dir /home/axh5735/projects/MPIR/MPIR-2.0/logs/codex_logs/codex_gray_colon_128_256/coords \
    --coords_read_dir /home/axh5735/projects/MPIR/MPIR-2.0/logs/codex_logs/codex_gray_colon_128_256/coords \
    || fail "python fail" \
    /
