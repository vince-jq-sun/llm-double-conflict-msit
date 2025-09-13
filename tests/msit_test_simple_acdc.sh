#!/bin/bash
# Simple ACDC test that works with current setup
# This runs the basic MSIT test first, then you can add ACDC later

echo "=== Running MSIT Test (ACDC-ready) ==="
python scripts_test/msit_api_test.py \
    --sessions 5 \
    --ndigits 3 \
    --stim_types "0,1,2,3" \
    --nrep 2 \
    --model "gpt2" \
    --restriction "strict" \
    --nickname "ready-for-acdc" \
    --output_dir "data/msit_pilot_outputs_smallnrep" \
    --randomized \
    --enable_acdc \
    --acdc_tau 1e-3 \
    --acdc_k_edges 50


echo ""
echo "=== Test Complete ==="
echo "To add ACDC analysis later, rerun with:"
echo "  --enable_acdc --acdc_tau 1e-3 --acdc_k_edges 50"
