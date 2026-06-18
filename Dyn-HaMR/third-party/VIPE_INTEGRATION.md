# VIPE Integration for Dyn-HaMR

This document explains how to use VIPE camera estimation instead of DROID-SLAM in the Dyn-HaMR pipeline.

## Quick Start (TL;DR)

**Step 1: Run VIPE on your video**
```bash
conda activate vipe
cd /data/home/zy3023/code/hand/Dyn-HaMR/third-party/vipe
vipe infer /data/home/zy3023/code/hand/Dyn-HaMR/test/videos/prod1.mp4
```

**Step 2: Run Dyn-HaMR with VIPE cameras**
```bash
conda activate slahmr
cd /data/home/zy3023/code/hand/Dyn-HaMR/dyn-hamr
python run_opt.py data=video_vipe data.seq=prod1
```

**Important Notes:**
- If VIPE results don't exist, the pipeline automatically falls back to DROID-SLAM
- If you previously ran with DROID-SLAM, use `overwrite=True` to regenerate cameras:
  ```bash
  python run_opt.py data=video_vipe data.seq=prod1 overwrite=True
  ```

**To verify the integration works:**
```bash
cd /data/home/zy3023/code/hand/Dyn-HaMR
python3 test_vipe_integration.py
```

## Overview

VIPE (Visual Inertial Pose Estimation) is an alternative camera estimation method that can be used instead of DROID-SLAM. The integration allows seamless switching between the two methods via configuration.

## VIPE Output Format

VIPE produces the following outputs:
- `pose/{sequence}.npz`: Camera-to-world transformation matrices
  - `data`: (N, 4, 4) c2w matrices
  - `inds`: (N,) frame indices
- `intrinsics/{sequence}.npz`: Camera intrinsics
  - `data`: (N, 4) [fx, fy, cx, cy]
  - `inds`: (N,) frame indices

## How to Use VIPE

### Option 1: Using the VIPE Config File

The easiest way is to use the pre-configured VIPE config:

```bash
cd dyn-hamr
python run_opt.py data=video_vipe data.seq=prod1
```

### Option 2: Override Config Parameters

You can override any existing config to use VIPE:

```bash
cd dyn-hamr
python run_opt.py data=video \
    data.seq=prod1 \
    data.use_vipe=True \
    data.vipe_dir=/path/to/vipe/results
```

### Option 3: Modify Your Config File

Add these lines to your data config YAML file:

```yaml
# VIPE-specific options
use_vipe: True
vipe_dir: /path/to/vipe/vipe_results
```

## Configuration Parameters

- `use_vipe` (bool): Set to `True` to use VIPE instead of DROID-SLAM
- `vipe_dir` (str): Path to VIPE results directory containing `pose/` and `intrinsics/` subdirectories

## How It Works

When `use_vipe=True`:

1. The pipeline loads VIPE camera poses and intrinsics from `.npz` files
2. Converts camera-to-world (c2w) matrices to world-to-camera (w2c) format
3. Saves cameras in DROID-SLAM compatible format (`cameras.npz`)
4. The rest of the Dyn-HaMR pipeline proceeds normally

The conversion happens automatically in `dyn-hamr/data/vidproc.py`:
- `load_vipe_cameras()`: Loads and converts VIPE outputs
- `save_vipe_cameras_as_droid()`: Saves in DROID-SLAM format
- `preprocess_cameras()`: Routes to VIPE or DROID-SLAM based on config

## Directory Structure

```
your_project/
├── third-party/
│   └── vipe/
│       └── vipe_results/
│           ├── pose/
│           │   └── prod1.npz
│           ├── intrinsics/
│           │   └── prod1.npz
│           └── rgb/
│               └── prod1.mp4
├── test/
│   ├── images/
│   │   └── prod1/
│   │       └── *.jpg
│   └── dynhamr/
│       └── cameras/
│           └── prod1/
│               └── shot-0/
│                   └── cameras.npz  # Generated from VIPE
```

## Example: Complete Workflow

1. **Run VIPE on your video:**
   ```bash
   # Activate VIPE environment and run camera estimation
   conda activate vipe
   cd /data/home/zy3023/code/hand/Dyn-HaMR/third-party/vipe
   vipe infer /data/home/zy3023/code/hand/Dyn-HaMR/test/videos/prod1.mp4
   
   # This generates:
   # - vipe_results/pose/prod1.npz
   # - vipe_results/intrinsics/prod1.npz
   # - vipe_results/rgb/prod1.mp4
   ```

2. **Verify VIPE results:**
   ```bash
   ls third-party/vipe/vipe_results/pose/prod1.npz
   ls third-party/vipe/vipe_results/intrinsics/prod1.npz
   ```

3. **Run Dyn-HaMR with VIPE:**
   ```bash
   conda activate slahmr  # or your Dyn-HaMR environment
   cd dyn-hamr
   python run_opt.py data=video_vipe data.seq=prod1
   ```

4. **The pipeline will:**
   - Extract frames from video
   - Run HaMeR for hand tracking
   - Load VIPE cameras (instead of running DROID-SLAM)
   - Optimize hand poses
   - Generate visualizations

## Running VIPE on New Videos

To use VIPE with a new video sequence:

```bash
# 1. Run VIPE
conda activate vipe
cd /data/home/zy3023/code/hand/Dyn-HaMR/third-party/vipe
vipe infer /data/home/zy3023/code/hand/Dyn-HaMR/test/videos/YOUR_VIDEO.mp4

# 2. Run Dyn-HaMR (it will automatically use VIPE if results exist)
conda activate slahmr
cd /data/home/zy3023/code/hand/Dyn-HaMR/dyn-hamr
python run_opt.py data=video_vipe data.seq=YOUR_VIDEO
```

**Note:** If VIPE results don't exist for a sequence, the pipeline will automatically fall back to DROID-SLAM with a warning message

## Verification

Run the test suite to verify the integration:

```bash
python3 test_vipe_integration.py
```

Expected output:
```
✓ All tests passed! VIPE integration is working correctly.
```

## Troubleshooting

### Error: "VIPE pose file not found"
- Ensure `vipe_dir` points to the correct directory
- Check that `{sequence}.npz` exists in `pose/` and `intrinsics/` subdirectories
- Verify the sequence name matches between config and VIPE outputs

### Error: "No images found"
- The pipeline will try to infer image size from intrinsics if images aren't available yet
- For best results, ensure frames are extracted before camera processing

### Cameras look wrong / VIPE not being used
- **Most common issue:** Old DROID-SLAM cameras still in the directory
- **Solution:** Delete the cameras directory or run with `overwrite=True`
- Verify VIPE cameras are loaded by checking the first camera matrix
- VIPE uses camera-to-world (c2w) format, which is automatically converted to world-to-camera (w2c)
- Check that VIPE's coordinate system matches your expectations
- Verify intrinsics are correct (fx, fy, cx, cy)

## Switching Back to DROID-SLAM

To use DROID-SLAM instead of VIPE:

```bash
cd dyn-hamr
python run_opt.py data=video data.seq=prod1 data.use_vipe=False
```

Or simply use a config without `use_vipe` set (defaults to DROID-SLAM).

## Technical Details

### Coordinate System Conversion

- **VIPE output**: Camera-to-world (c2w) matrices
- **Dyn-HaMR expects**: World-to-camera (w2c) matrices
- **Conversion**: `w2c = inv(c2w)`

### Intrinsics Format

- **VIPE**: (N, 4) [fx, fy, cx, cy]
- **Dyn-HaMR**: (N, 6) [fx, fy, cx, cy, W, H]
- **Conversion**: Image dimensions are added from actual images or inferred from intrinsics

### Saved Format

The `cameras.npz` file contains:
- `w2c`: (N, 4, 4) world-to-camera matrices
- `intrins`: (N, 4) [fx, fy, cx, cy]
- `focal`: float, average of fx and fy
- `width`: int, image width
- `height`: int, image height

This format is identical to DROID-SLAM output, ensuring compatibility with the rest of the pipeline.

