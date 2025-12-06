<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)
![LLM](https://img.shields.io/badge/LLM-Vertex_AI_Gemini_2.5-orange?logo=google&logoColor=white)
![ML](https://img.shields.io/badge/ML-HuggingFace-yellow?logo=huggingface&logoColor=white)
![VectorDB](https://img.shields.io/badge/VectorDB-ChromaDB-purple?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PC9zdmc+)
![License](https://img.shields.io/badge/License-MIT-green)

# MedSaC

### Medical Stepback and Calculation Benchmark

**Evaluating Step-Back prompting on clinical calculation tasks from patient notes**

---

</div>

## Overview

Medical calculations are critical in clinical practice, from estimating kidney function (GFR, CrCl) to assessing stroke risk (CHA2DS2-VASc). This benchmark evaluates how well LLMs can:

1. Extract relevant values from patient notes
2. Identify the correct formula to use
3. Perform accurate calculations
4. Provide clinically valid results

## Features

- **Multiple Prompting Methods**: Compare Direct, Chain-of-Thought (CoT), Step-Back, and MedRaC-style prompting
- **RAG Support**: Retrieval-augmented generation using ChromaDB + HuggingFace embeddings for formula lookup
- **Code Execution**: MedRaC-style methods generate Python code for calculations (avoids LLM arithmetic errors)
- **Comprehensive Evaluation**: Regex-based and LLM-based evaluators
- **Error Analysis**: Detailed error type classification (formula errors, extraction errors, arithmetic errors, etc.)
- **Visualization**: Built-in charts for accuracy comparison, error analysis, and performance metrics
- **Vertex AI Integration**: Uses Google's Gemini 2.5 models via Vertex AI

## Project Structure

```
MedSaC/
├── data/                    # Input datasets
│   ├── test_data.csv        # Patient notes and ground truth
│   ├── formula_new.json     # Medical formulas by Calculator ID
│   └── web_formula.txt      # Formula knowledge base for RAG
├── method/                  # Prompting methods
│   ├── method.py            # Base class with prompt templates
│   ├── plain.py             # Plain wrapper with code execution
│   └── rag.py               # RAG utility (ChromaDB + HuggingFace)
├── model/                   # LLM integrations
│   └── vertexai.py          # Vertex AI / Gemini client
├── evaluator/               # Evaluation modules
│   ├── regEvaluator.py      # Regex-based evaluator
│   └── llmEvaluator.py      # LLM-based evaluator
├── schema/                  # Pydantic output schemas
│   └── schemas.py           # Structured output definitions
├── utils/                   # Utilities
│   └── error_type.py        # Error type analysis
├── run.py                   # Main entry point
├── visualize_results.py     # Results visualization
└── requirements.txt         # Dependencies
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/coolgenerator/MedSaC.git
cd MedSaC
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up Vertex AI credentials:
   - Create a service account in Google Cloud Console
   - Download the JSON key file
   - Save it as `model/vertex-ai-credential.json`

## Usage

### Basic Usage

Edit `run.py` to select your method and run:

```bash
python run.py
```

### Available Methods

| Method | Description | RAG | Code Execution |
|--------|-------------|-----|----------------|
| `direct` | Direct answer prompting | No | No |
| `cot` | Chain-of-Thought reasoning | No | No |
| `stepback` | Step-Back prompting | No | No |
| `direct_rag` | Direct with formula retrieval | Yes | No |
| `cot_rag` | CoT with formula retrieval | Yes | No |
| `stepback_rag` | Step-Back with formula retrieval | Yes | No |
| `stepback_calc_rag` | Step-Back + Python code execution | Yes | Yes |
| `medrac_rag` | MedRaC-style (extract values + code) | Yes | Yes |

**Example: RAG-enhanced Step-Back with Code Execution**
```python
from method.rag import RAG

# Initialize RAG
rag = RAG(
    doc_path="data/web_formula.txt",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    embeddings_dir="data/formula_embeddings_chroma",
)

# Create method
method = Plain(
    "stepback_calc_rag",  # or "medrac_rag"
    [gemini_gen],
    [reg_evaluator, llm_evaluator],
    rag=rag,
)
```

### Running Evaluation

```python
# Generate raw outputs
raw = method.generate_raw(test=True)  # test=False for full dataset

# Evaluate results
eval_json = method.evaluate(raw_json_file=raw)

# Compute accuracy statistics
reg_evaluator.compute_overall_accuracy_new(
    input_file_path=eval_json,
    output_dir_path="stats"
)

# Run error type analysis
error_type_pipeline(
    input_json=eval_json,
    output_json_dir="ErrorTypes",
    model=gemini_eval
)
```

### Visualization

Generate comparison charts from evaluation results:

```bash
python visualize_results.py
```

This creates:
- Overall accuracy comparison across methods
- Per-category accuracy breakdown
- Error type distribution
- Precision/Recall/F1 metrics

## Medical Calculators Included

The benchmark includes various clinical calculators:

- **Renal Function**: Cockcroft-Gault CrCl, CKD-EPI GFR, MDRD GFR
- **Cardiovascular Risk**: CHA2DS2-VASc, Wells' Criteria for PE/DVT
- **Basic Measurements**: BMI, MAP, Ideal Body Weight
- **Lab Corrections**: Calcium correction for hypoalbuminemia
- **And more...**

## Output

Results are saved to:
- `raw_output/` - Raw LLM responses
- `eval_output/` - Evaluation results with correctness labels
- `stats/` - Accuracy statistics (CSV)
- `ErrorTypes/` - Detailed error analysis
- `visualizations/` - Generated charts

## Requirements

- Python 3.9+
- Google Cloud account with Vertex AI enabled
- ~500MB disk space for HuggingFace embeddings

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This project is based on [MedRaC](https://github.com/Super-Billy/EMNLP-2025-MedRaC/) (EMNLP 2025), licensed under Apache-2.0. We thank the original authors for their dataset and codebase.

**Changes from the original:**
- Replaced OpenAI/DeepSeek with Vertex AI (Gemini 2.5)
- Replaced OpenAI embeddings with HuggingFace embeddings for RAG
- Replaced FAISS with ChromaDB for vector storage
- Added `stepback_calc_rag` method (Step-Back + code execution)
- Added visualization script for results analysis
- Simplified to focus on Direct, CoT, Step-Back, and MedRaC prompting methods

**Data sources:**
- Patient notes sourced from PubMed Central (PMC) articles
- Medical formulas based on standard clinical calculators

## Citation

If you use this work, please cite the original MedRaC paper:

```bibtex
@inproceedings{medrac2025,
  title={MedRaC: Medical Reasoning and Calculation Benchmark},
  author={Super-Billy et al.},
  booktitle={EMNLP},
  year={2025}
}
```

*Note: Please check the [original repository](https://github.com/Super-Billy/EMNLP-2025-MedRaC/) for the official citation once the paper is published.*
