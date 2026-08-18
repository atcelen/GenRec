# Example inputs

Drop one or more images here to try the general inference script out of the box:

```bash
INPUT_DIR=assets/examples bash scripts/run_inference.sh
```

- **One image** → a synthesized camera trajectory around it
  (`TRAJECTORY=orbit|dolly|spiral|wiggle`).
- **Two or more images** → SLERP interpolation between the input viewpoints.
- Pass `POSES=path/to/poses.json` to render specific camera poses instead.

The committed `example_*.png` images are synthetic **placeholders** so the
quick-start command runs end to end — swap in real photos for meaningful
results. (TODO(release): replace with 1–2 real sample photos before announcing.)
