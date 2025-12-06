from pydantic import BaseModel, Field
from typing import List

# Used for extracted_values in structured output
class KeyValue(BaseModel):
    key: str = Field(..., description="The variable name")
    value: str = Field(..., description="The value extracted for the variable")

def prompt_style_to_schema(prompt_style: str) -> BaseModel:
    if prompt_style == 'direct':
        return DirectOutput
    elif prompt_style == 'direct_rag':
        return DirectOutput
    elif prompt_style == 'cot':
        return CoTOutput
    elif prompt_style == 'cot_rag':
        return CoTOutput
    elif prompt_style == 'stepback':
        return CoTOutput  # Uses same output format as CoT
    elif prompt_style == 'stepback_rag':
        return CoTOutput
    elif prompt_style == 'stepback_calc_rag':
        return StepBackCalcOutput  # Includes python_code
    elif prompt_style == 'medrac_rag':
        return MedRaCOutput  # MedRaC style output
    elif prompt_style == 'oneshot':
        return OneShotOutput
    elif prompt_style == 'modular':
        return ModularOutput
    elif prompt_style == 'modular_cot':
        return ModularCoTOutput
    else:
        print(f"Prompt style: {prompt_style} doesn't have a schema yet.")
        return None  # No schema for this prompt style

class DirectOutput(BaseModel):
    answer: str = Field(..., description="Short and direct answer to the question")

class CoTOutput(BaseModel):
    step_by_step_thinking: str = Field(..., description="Step-by-step reasoning to reach the answer")
    answer: str = Field(..., description="Final answer after reasoning")

OneShotOutput = CoTOutput

class ModularOutput(BaseModel):
    formula: str = Field(..., description="Explicit formula used for the calculation")
    extracted_values: List[KeyValue] = Field(..., description="List of extracted variable names and values from the patient note")
    answer: str = Field(..., description="Final answer calculated using the formula")

class ModularCoTOutput(BaseModel):
    formula_reason: str = Field(..., description="Reasoning behind choosing the formula")
    formula: str = Field(..., description="The explicit formula used for calculation")
    extracted_values_reason: str = Field(..., description="Explanation of how each value was identified")
    extracted_values: List[KeyValue] = Field(..., description="List of extracted variable names and values")
    calculation_steps: str = Field(..., description="Step-by-step breakdown of the calculation")
    answer: str = Field(..., description="Final result of the calculation")

class StepBackCalcOutput(BaseModel):
    """Output schema for stepback_calc_rag - StepBack + Code execution"""
    step_by_step_thinking: str = Field(..., description="Step-by-step reasoning including formula verification")
    extracted_values: dict = Field(..., description="Dictionary of extracted variable names and values")
    python_code: str = Field(..., description="Python code to compute the result, storing answer in 'result' variable")
    answer: str = Field(..., description="The computed answer from executing the code")

class MedRaCOutput(BaseModel):
    """Output schema for medrac_rag - MedRaC style with code execution"""
    extracted_values: dict = Field(..., description="Dictionary of extracted variable names and values")
    python_code: str = Field(..., description="Python code to compute the result, storing answer in 'result' variable")

class Values(BaseModel):
    """Schema for extracted values only (used with RAG where formula is provided)"""
    extracted_values_reason: str = Field(..., description="Explanation of how values were identified")
    extracted_values: dict = Field(..., description="Dictionary of variable names to extracted values")

class FormulaAndValues(BaseModel):
    """Schema for both formula and extracted values (used without RAG)"""
    formula_reason: str = Field(..., description="Reasoning for choosing this formula")
    formula: str = Field(..., description="The mathematical formula to apply")
    extracted_values_reason: str = Field(..., description="Explanation of how values were identified")
    extracted_values: dict = Field(..., description="Dictionary of variable names to extracted values")

def get_schema(field_name: str) -> BaseModel:
    if field_name == "formula":
        return FormulaEvaluation
    elif field_name == "extracted_values":
        return VariableEvaluation
    elif field_name == "calculation":
        return CalculationEvaluation
    elif field_name == "answer":
        return FinalansEvaluation
    else:
        print(f"Field name: {field_name} doesn't have a schema yet.")
        return None

class EvaluationAspect(BaseModel):
    result: str = Field(..., description='The binary evaluation result', json_schema_extra={"enum":["Correct", "Incorrect"]})
    explanation: str = Field( ...,description="A brief explanation for the evaluation.")
    

class FormulaEvaluation(BaseModel):
    formula: EvaluationAspect = Field(...,description="Evaluation for the formula or scoring criteria.")
    class Config:
        extra = "forbid"

class VariableEvaluation(BaseModel):
    extracted_values: EvaluationAspect = Field( ...,description="Evaluation for the variable substitution.")
    class Config:
        extra = "forbid"

class CalculationEvaluation(BaseModel):
    calculation: EvaluationAspect = Field(...,description="Evaluation for the calculation process.")
    class Config:
        extra = "forbid"

class FinalansEvaluation(BaseModel):
    answer: EvaluationAspect = Field( ...,description="Evaluation for the final answer.")
    class Config:
        extra = "forbid"

