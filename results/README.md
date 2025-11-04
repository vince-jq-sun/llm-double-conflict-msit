# Results Directory

This directory contains MSIT test results and analysis outputs.

## Structure

- `graphs/` - Analysis plots and visualizations
- `msit_pilot_figures/` - Pilot study results and figures
- Test result folders are created automatically with timestamps

## Generated Files

Test runs create folders with the format:
```
YYYYMMDD_HHMMSS_model-{model_name}_sessions-{num_sessions}/
├── session_1.json
├── session_2.json
├── ...
└── summary.json
```

Each session file contains:
- Stimulus configurations
- Model responses
- Extracted answers
- Timing information
- Metadata

## Analysis

Use the scripts in `scripts_analysis/` to process results:

```bash
# Analyze accuracy
python scripts_analysis/analyze_msit_results.py results/your_test_folder

# Error classification
python scripts_analysis/analyze_error_types.py results/your_test_folder

# Generate visualizations
python scripts_analysis/plot_msit_heatmap.py results/your_test_folder
```
