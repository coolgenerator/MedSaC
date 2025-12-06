#!/usr/bin/env python3
"""
run.py

A demo file for comparing Direct, CoT, and StepBack methods with RAG.
"""
import sys
import random
import pandas as pd
sys.set_int_max_str_digits(0)

from model import APIModel, vllmModels
from evaluator import RegEvaluator, LLM_Evaluator
from method.plain import Plain
from method.rag import RAG
from utils.error_type import error_type_pipeline


# Models: Split quota (1000 RPM total) between generation and evaluation
# Gemini 2.5 Flash - for answer generation
gemini_gen = APIModel(
    'VertexAI/gemini-2.5-flash',
    rpm_limit=500,
    tpm_limit=2000000,
    temperature=1.0,
    disable_thinking=True,  # Disable thinking
)

# Gemini 2.5 Pro - for evaluations (best reasoning, with thinking enabled)
gemini_eval = APIModel(
    'VertexAI/gemini-2.5-pro',
    rpm_limit=500,
    tpm_limit=2000000,
    temperature=0.0,
    disable_thinking=False,  # Keep thinking enabled for better evaluation
)

llm_evaluator = LLM_Evaluator(gemini_eval)  # Use 2.5 Pro for LLM evaluation
reg_evaluator = RegEvaluator()

# Initialize RAG for formula retrieval
print("Initializing RAG for formula retrieval...")
rag = RAG(
    doc_path="data/web_formula.txt",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    embeddings_dir="data/formula_embeddings_chroma",
)
print(f"RAG initialized with {len(rag)} formula blocks")

# If you want to use open-source models, uncomment and replace gemini with model
# model = vllmModels(model_name="meta-llama/Llama-3.1-8B-Instruct")


# Available methods: Direct, CoT (Chain-of-Thought), StepBack

# ------- Direct Method ------
# method = Plain(
#     "direct",
#     [gemini],
#     [reg_evaluator, llm_evaluator]
# )
# raw = method.generate_raw(test=True)
# eval_json = method.evaluate(raw_json_file=raw)
# reg_evaluator.compute_overall_accuracy_new(input_file_path=eval_json, output_dir_path="stats")

# ------- CoT (Chain-of-Thought) Method ------
# method = Plain(
#     "cot",
#     [gemini],
#     [reg_evaluator, llm_evaluator]
# )
# raw = method.generate_raw(test=True)
# eval_json = method.evaluate(raw_json_file=raw)
# reg_evaluator.compute_overall_accuracy_new(input_file_path=eval_json, output_dir_path="stats")

# ------- StepBack Method ------
# method = Plain(
#     "stepback",
#     [gemini],
#     [reg_evaluator, llm_evaluator]
# )
# raw = method.generate_raw(test=True)
# eval_json = method.evaluate(raw_json_file=raw)
# reg_evaluator.compute_overall_accuracy_new(input_file_path=eval_json, output_dir_path="stats")

# ------------ Error Type Analysis -------------
# error_type_pipeline(input_json=eval_json, output_json_dir="ErrorTypes", model_name='VertexAI/gemini-2.5-flash')


# ------------ Evaluate all methods -------------
# Priority indices: questions that LLMs often get wrong
hard_questions = [21, 28, 32, 39, 41, 43, 48, 49, 51, 55, 56, 88]

# Generate random samples ONCE so all methods use the same data

random.seed(60)  # For reproducibility
df = pd.read_csv('./data/test_data.csv')
n_random = 50
random_from_index = 200

# Get random indices (excluding hard_questions)
available_indices = [i for i in range(random_from_index, len(df)) if i not in hard_questions]
random_indices = random.sample(available_indices, min(n_random, len(available_indices)))

# Combine: hard questions + random samples
all_test_indices = list(hard_questions) + random_indices
print(f"Test data: {len(hard_questions)} hard questions + {len(random_indices)} random samples = {len(all_test_indices)} total")
print(f"Test indices: {all_test_indices}")

# RAG-enhanced methods for all three prompting strategies
methods = [ "cot_rag", "stepback_calc_rag", "medrac_rag", "direct_rag"]

for style in methods:
    print(f"\n{'='*60}")
    print(f"  METHOD: {style.upper()}")
    print(f"{'='*60}")

    method = Plain(
        style,
        [gemini_gen],                     # Use 2.5 Flash for generation
        [reg_evaluator, llm_evaluator],   # LLM evaluator uses 2.5 Pro
        rag=rag,                          # Pass RAG instance for formula retrieval
        row_numbers=all_test_indices,     # Use the SAME indices for all methods
    )

    # Stage 1: Generate raw outputs
    print(f"\n[STAGE 1/4] Generating raw outputs ({style})...")
    raw = method.generate_raw(test=True)

    # Stage 2: Evaluate results
    print(f"\n[STAGE 2/4] Running LLM evaluation ({style})...")
    eval_json = method.evaluate(raw_json_file=raw)

    # Stage 3: Compute accuracy statistics
    print(f"\n[STAGE 3/4] Computing accuracy statistics ({style})...")
    reg_evaluator.compute_overall_accuracy_new(
        input_file_path=eval_json,
        output_dir_path="stats"
    )

    # Stage 4: Error type analysis (reuses gemini_eval model)
    print(f"\n[STAGE 4/4] Running error type analysis ({style})...")
    error_type_pipeline(
        input_json=eval_json,
        output_json_dir="ErrorTypes",
        model=gemini_eval  # Use 2.5 Pro for error analysis
    )

    print(f"\n✓ Completed {style.upper()} method")
