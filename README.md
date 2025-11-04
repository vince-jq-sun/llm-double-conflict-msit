# MSIT-LLM: Multi-Source Interference Task for Language Models

A research framework for testing cognitive interference patterns in Large Language Models using the Multi-Source Interference Task (MSIT) paradigm.

## Overview

This project implements a comprehensive testing framework to evaluate how language models handle cognitive interference similar to human psychological experiments. The MSIT paradigm tests spatial (Simon) and flanker interference effects across different stimulus conditions.

## Project Structure

```
├── scripts_test/           # Core testing scripts
│   ├── msit_gen.py        # MSIT stimulus generator (11 condition types)
│   ├── msit_api_test.py   # Main API testing script
│   ├── local_model_handler.py  # Local model support (GPT-2, LLaMA)
│   └── ollama_chat.py     # Ollama integration
├── scripts_analysis/      # Results analysis tools
│   ├── analyze_msit_results.py    # Accuracy analysis
│   ├── analyze_error_types.py     # Error classification
│   └── plot_msit_heatmap.py       # Visualization tools
├── results/               # Generated results and figures
│   ├── graphs/           # Analysis plots
│   └── msit_pilot_figures/  # Pilot study results
├── tests/                # Automated test scripts
├── ollama_llama_caller.py # Standalone Ollama interface
└── requirements_local.txt # Python dependencies
```

## MSIT Conditions

The framework supports 11 different stimulus conditions:

| Code | Condition | Description | Example |
|------|-----------|-------------|---------|
| 0 | Cg | Congruent | `100` (target=1, position=1) |
| 1 | Sm | Simon-only | `010` (target=1, position=2) |
| 2 | Fk | Flanker-only | `122` (target=1, flankers=2) |
| 3 | SmFk | Simon+Flanker | `221` (target=1, pos=3, flankers=2) |
| 4 | CgLtr | Congruent Letter | `t00` (letter target) |
| 5 | CgExN | Congruent Extra Number | `600` (extended digits) |
| 6 | CgExN-R | Restricted Extra Number | `400` (limited variants) |
| 7 | SmExNFk | Simon+Extra+Flanker | `115` |
| 8 | SmFkExN | Simon+Flanker+Extra | `552` |
| 9 | SmFkIdPos | Simon+Flanker+Identity | `331` |
| 10 | CgFkExN | Congruent+Flanker+Extra | `525` |

## Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements_local.txt

# For API testing (optional)
pip install openai anthropic requests
```

### 2. Basic Usage

#### Generate MSIT Stimuli
```bash
python scripts_test/msit_gen.py --conditions 0,1,2,3 --repetitions 10
```

#### Test with OpenAI API
```bash
python scripts_test/msit_api_test.py --model gpt-4 --sessions 5 --conditions 0,1,2,3
```

#### Test with Local Models
```bash
python scripts_test/msit_api_test.py --model gpt2 --sessions 3 --local
```

#### Test with Ollama
```bash
# Start Ollama server first
ollama serve

# Run tests
python scripts_test/msit_api_test.py --model llama3.1:8b-instruct-q4_0 --ollama --sessions 5
```

### 3. Analysis

#### Analyze Results
```bash
python scripts_analysis/analyze_msit_results.py results/your_test_folder
```

#### Error Type Analysis
```bash
python scripts_analysis/analyze_error_types.py results/your_test_folder
```

#### Generate Heatmaps
```bash
python scripts_analysis/plot_msit_heatmap.py results/your_test_folder
```

## Supported Models

### API Models
- **OpenAI**: GPT-4, GPT-3.5-turbo, GPT-4-turbo
- **Anthropic**: Claude-3 (Sonnet, Haiku, Opus)
- **Local APIs**: Any OpenAI-compatible endpoint

### Local Models
- **Hugging Face**: GPT-2, LLaMA, Mistral, etc.
- **Ollama**: LLaMA 3.1, Mistral, CodeLlama, etc.

## Configuration

### Environment Variables
```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
```

### Model Configuration
Edit model parameters in the test scripts:
- `temperature`: Sampling temperature (0.0-1.0)
- `max_tokens`: Maximum response length
- `repetitions`: Trials per condition

## Research Applications

This framework enables research into:

1. **Cognitive Interference**: How LLMs handle conflicting spatial and semantic information
2. **Attention Mechanisms**: Selective attention patterns in transformer models  
3. **Error Analysis**: Classification of response errors (Simon, Flanker, Other)
4. **Model Comparison**: Systematic evaluation across different architectures
5. **Scaling Effects**: Performance patterns across model sizes

## Output Format

Results are saved as JSON files containing:
- Stimulus configurations and correct answers
- Model responses and extracted answers
- Timing information and metadata
- Error classifications and accuracy metrics

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{msit_llm_2024,
  title={MSIT-LLM: Multi-Source Interference Task for Language Models},
  author={[Your Name]},
  year={2024},
  url={https://github.com/your-username/llm-double-conflict-msit}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
