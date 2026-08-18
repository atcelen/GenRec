"""
RealEstate10K evaluation — data processing + metrics.

Thin entry point over the shared evaluation engine (genrec/cli/evaluate_ours.py). It fixes
the dataset to RealEstate10K and applies the RE10K evaluation defaults; everything
else is forwarded to the engine and can be overridden on the command line.

The RE10K dataset is read by genrec.data.dataset.RealEstate10KDatasetWithEulerRaw
(one torch/.pt file per scene). Point --data_path at the directory of per-scene files.

Usage (run on a GPU compute node):

    python -m genrec.cli.eval_re10k \\
        --data_path /path/to/RealEstate10K \\
        --checkpoint genrec \\
        --output_dir ./re10k_eval_results
"""

from genrec.cli.evaluate_ours import run_eval

# RE10K evaluation protocol defaults (overridable on the CLI).
RE10K_DEFAULTS = {
    "config": "configs/re10k.yaml",
    "output_dir": "./re10k_eval_results",
    "scene_list": "assets/re10k_test_scenes_100.txt",
    "step_size": 10,
    "num_steps": 25,
    "guidance": 1.5,
    "use_gt_poses": True,
    "obs_masks_dir": "paper",
    "save_visualizations": True,
}


def main():
    run_eval("re10k", defaults=RE10K_DEFAULTS)


if __name__ == "__main__":
    main()
