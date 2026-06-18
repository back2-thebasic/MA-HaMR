#!/usr/bin/env bash
# Convert dev-stage H2O + HOT3D sequences to Dyn-HaMR mp4 inputs.
set -euo pipefail

VIDEO_DIR="/extra/SuC/dynhamr_io/videos"
MAHMR="/data/SuC/MA-HaMR"

mkdir -p "$VIDEO_DIR"

echo "========== HOT3D =========="
conda run -n hot3d python "$MAHMR/scripts/convert_hot3d_to_mp4.py" \
  --sequence-dir /extra/SuC/data/raw/hot3d/dataset/P0003_c701bd11 \
  --output "$VIDEO_DIR/hot3d_P0003.mp4" \
  --hot3d-repo /extra/SuC/data/raw/hot3d

echo ""
echo "========== H2O (train dev clips) =========="
# subject1: h1, h2 ; subject2: h1
for spec in "1 h1" "1 h2" "2 h1"; do
  set -- $spec
  subj=$1
  act=$2
  conda run -n dynhamr python "$MAHMR/scripts/convert_h2o_to_mp4.py" \
    --subject "$subj" \
    --action "$act" \
    --take 0 \
    --rgb-kind rgb \
    --fps 30 \
    --output "$VIDEO_DIR/h2o_s${subj}_${act}.mp4"
done

echo ""
echo "========== Done. Videos in $VIDEO_DIR =========="
ls -lh "$VIDEO_DIR"/hot3d_P0003.mp4 "$VIDEO_DIR"/h2o_s1_h1.mp4 "$VIDEO_DIR"/h2o_s1_h2.mp4 "$VIDEO_DIR"/h2o_s2_h1.mp4
