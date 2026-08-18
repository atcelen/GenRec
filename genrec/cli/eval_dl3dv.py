"""
DL3DV-10K evaluation — data processing + metrics.

Thin entry point over the shared evaluation engine (genrec/cli/evaluate_ours.py). It fixes
the dataset to DL3DV-10K and applies the DL3DV evaluation defaults; everything else
is forwarded to the engine and can be overridden on the command line.

The DL3DV dataset is read by genrec.data.dataset.DL3DVDataset (one .h5 file per
scene). Point --data_path at the directory of per-scene files.

Usage (run on a GPU compute node):

    python -m genrec.cli.eval_dl3dv \\
        --data_path /path/to/DL3DV-10K \\
        --checkpoint genrec \\
        --output_dir ./dl3dv_eval_results
"""

from genrec.cli.evaluate_ours import run_eval

# DL3DV-10K evaluation protocol defaults (overridable on the CLI).
DL3DV_DEFAULTS = {
    "config": "configs/dl3dv.yaml",
    "output_dir": "./dl3dv_eval_results",
    "scene_list": "assets/dl3dv_test_scenes_100.txt",
    "longest_side": 616,
    "step_size": 4,
    "num_steps": 15,
    "guidance": 1.5,
    "use_gt_poses": True,
    "obs_masks_dir": "paper",
    "save_visualizations": True,
    "save_images": True,
}


def main():
    run_eval("dl3dv", defaults=DL3DV_DEFAULTS)


if __name__ == "__main__":
    main()
