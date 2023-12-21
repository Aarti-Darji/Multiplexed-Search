#!/bin/bash

#SBATCH --job-name=1codex_Tonsil_0_128_256
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=120:00:00
#SBATCH --error=/home/acd9300/projects/project2/compressed_images_codex/codexlogs/MPIR/tonsil_logs/codex_Tonsil_0_128_256/1codex_Tonsil_0_128_256.err
#SBATCH --output=/home/acd9300/projects/project2/compressed_images_codex/codexlogs/MPIR/tonsil_logs/codex_Tonsil_0_128_256/1codex_Tonsil_0_128_256.out




function fail {
    echo "FAIL: $@" >&2
    exit 1  # signal failure
}

source /home/acd9300/miniconda3/etc/profile.d/conda.sh || fail "conda load fail"
conda activate MPIR || fail "conda activate fail"

cd '/home/acd9300/projects/project2/MPIR-2.0/codex_Model_updated'


python '/home/acd9300/projects/project2/MPIR-2.0/codex_Model_updated/runfile.py' \
    --transformations_write_dir "/home/acd9300/projects/project2/compressed_images_codex/codexlogs/MPIR/tonsil_logs/codex_Tonsil_0_128_256/transformations" \
    --inv_transformations_read_dir "/home/acd9300/projects/project2/compressed_images_codex/codexlogs/MPIR/tonsil_logs/codex_Tonsil_0_128_256/transformations" \
    --transformations_read_dir "/home/acd9300/projects/project2/compressed_images_codex/codexlogs/MPIR/tonsil_logs/codex_Tonsil_0_128_256/transformations" \
    --patch_size 64 \
    --num_patches_per_image 200 \
    --selected_channel 0 \
    --whitespace_threshold 0.82 \
    --test_ratio 0.2 \
    --val_ratio 0.2 \
    --everything_seed 1 \
    --split_seed 2 \
    --shuffling_seed 3 \
    --patching_seed 4 \
    --logging_dir "/home/acd9300/projects/project2/compressed_images_codex/codexlogs/MPIR/tonsil_logs/codex_Tonsil_0_128_256/tb_logs" \
    --logging_name codex_Tonsil_0_128_256 \
    --batch_size 128 \
    --num_dataloader_workers 8 \
    --normalize_transform \
    --input_height 64 \
    --latent_dim 256 \
    --lr 0.0001 \
    --gradient_clip_val 0.5 \
    --num_nodes 1 \
    --max_epochs 1 \
    --devices 1 \
    --accelerator gpu \
    --strategy ddp \
    --auto_lr_find False \
    --auto_scale_batch_size False \
    --benchmark True \
    --fast_dev_run False \
    --coords_write_dir "/home/acd9300/projects/project2/compressed_images_codex/codexlogs/MPIR/tonsil_logs/codex_Tonsil_0_128_256/coords" \
    --coords_read_dir "/home/acd9300/projects/project2/compressed_images_codex/codexlogs/MPIR/tonsil_logs/codex_Tonsil_0_128_256/coords" \
    --prepare \
    || fail "python fail" \
    


