#!/usr/bin/env python3
"""
run.py

A demo file for comparing Direct, CoT, and StepBack methods.
"""
import sys
sys.set_int_max_str_digits(0)

from model import APIModel, vllmModels
from evaluator import RegEvaluator, LLM_Evaluator

from method.plain    import Plain
from method.stepBack import StepBack

from utils.error_type import error_type_pipeline


# Default model: Vertex AI gemini-2.5-flash
gemini = APIModel(
    'VertexAI/gemini-2.5-flash',
    rpm_limit=1000,
    tpm_limit=4000000,
    temperature=1.0,
)

llm_evaluator = LLM_Evaluator(gemini)
reg_evaluator = RegEvaluator()

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
method = Plain(
    "cot",
    [gemini],
    [reg_evaluator, llm_evaluator]
)
raw = method.generate_raw(test=True)
eval_json = method.evaluate(raw_json_file=raw)
reg_evaluator.compute_overall_accuracy_new(input_file_path=eval_json, output_dir_path="stats")

# ------- StepBack Method ------
# method = StepBack(
#     llms=[gemini],
#     evaluators=[reg_evaluator, llm_evaluator],
#     use_rag=False,  # Set True to use RAG for formula retrieval
# )
# raw = method.generate_raw(test=True)
# eval_json = method.evaluate(raw_json_file=raw)
# reg_evaluator.compute_overall_accuracy_new(input_file_path=eval_json, output_dir_path="stats")


# ------------ Error Type Analysis -------------
error_type_pipeline(input_json=eval_json, output_json_dir="ErrorTypes", model_name='VertexAI/gemini-2.5-flash')
