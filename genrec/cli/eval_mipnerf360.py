"""
MipNeRF360 evaluation — data processing + metrics.

Thin entry point over the shared evaluation engine (genrec/cli/evaluate_ours.py). It fixes
the dataset to MipNeRF360 and applies its evaluation defaults; everything else is
forwarded to the engine and can be overridden on the command line.

The MipNeRF360 dataset is read by genrec.data.dataset.MipNeRF360Dataset (per-scene
COLMAP sparse/0 + images_<factor>/). The engine evaluates all scenes (the scene list
is ignored for this dataset). Point --data_path at the MipNeRF360 root and pick the
downsampling --factor (1/2/4/8).

Usage (run on a GPU compute node):

    python -m genrec.cli.eval_mipnerf360 \\
        --data_path /path/to/mipnerf360 \\
        --checkpoint genrec \\
        --factor 4 \\
        --output_dir ./mipnerf360_eval_results
"""

from genrec.cli.evaluate_ours import run_eval

# MipNeRF360 evaluation protocol defaults (overridable on the CLI).
MIPNERF360_DEFAULTS = {
    "config": "configs/dl3dv.yaml",
    "output_dir": "./mipnerf360_eval_results",
    "num_steps": 15,
    "guidance": 1.5,
    "factor": 4,
    "use_gt_poses": True,
    "obs_masks_dir": "paper",
    "save_visualizations": True,
    "save_images": True,
}


def main():
    run_eval("mipnerf360", defaults=MIPNERF360_DEFAULTS)


if __name__ == "__main__":
    main()
