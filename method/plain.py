from typing import List, Dict
from method.method import Method
from evaluator import Evaluator
from model.model import LLM
from schema.schemas import prompt_style_to_schema

import os
import re
import json
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd  # new dependency for convenient table handling

logger = logging.getLogger(__name__)


def execute_code(code: str) -> str:
    """
    Execute Python code in a sandboxed environment and return the result.
    The code should store its answer in a variable called `result`.
    """
    # Strip markdown code fences if present
    code_str = code.strip()
    if code_str.startswith("```"):
        code_str = re.sub(r"^```[^\n]*\n", "", code_str)
        code_str = re.sub(r"```$", "", code_str)

    # Create a safe execution environment
    safe_globals = {
        "math": math,
        "np": np,
        "numpy": np,
        "__builtins__": {
            "abs": abs, "round": round, "min": min, "max": max,
            "sum": sum, "len": len, "float": float, "int": int,
            "str": str, "bool": bool, "list": list, "dict": dict,
            "range": range, "pow": pow,
        }
    }
    local_vars = {}

    try:
        exec(code_str, safe_globals, local_vars)
        result = local_vars.get("result", None)
        if result is not None:
            return str(result)
        return "No result variable found"
    except Exception as e:
        return f"Execution error: {e}"


class Plain(Method):
    """Plain prompting strategy wrapper.

    1. ``generate_raw`` - run the model, collect raw answers/token counts
       and write them to ``<raw_json_dir>/<model>_<prompt_style>_raw.json``.

    2. ``evaluate`` - read the raw file, run every Evaluator object and
       append their results to a new file
       ``<eval_json_dir>/<model>_<prompt_style>_eval.json``.

    This separation makes it possible to cache expensive model calls and
    to re-run evaluation logic without touching the LLM again.
    """

    def __init__(
        self,
        prompt_style: str,
        llms: List[LLM],
        evaluators: List[Evaluator],
        rag=None,  # Optional RAG instance for RAG-enhanced methods
        **kwargs,
    ) -> None:
        # Methods that don't require RAG
        valid_prompt_styles = {
            "direct": self.direct,
            "cot": self.cot,
            "stepback": self.stepback,
            "oneshot": self.one_shot,
            "modular": self.modular,
            "modular_cot": self.modular_cot,
        }
        # Methods that require RAG
        rag_prompt_styles = {
            "direct_rag": self.direct_rag,
            "cot_rag": self.cot_rag,
            "stepback_rag": self.stepback_rag,
            "stepback_calc_rag": self.stepback_calc_rag,  # StepBack + Code execution
            "medrac_rag": self.medrac_rag,  # MedRaC style with RAG
        }

        all_styles = {**valid_prompt_styles, **rag_prompt_styles}

        if prompt_style not in all_styles:
            raise ValueError(f"Prompt style: {prompt_style} not supported.")

        # Check if RAG is required but not provided
        if prompt_style in rag_prompt_styles and rag is None:
            raise ValueError(f"Prompt style '{prompt_style}' requires a RAG instance.")

        self.prompt_style = prompt_style
        self.prompt_fn = all_styles[prompt_style]
        self.evaluators = evaluators
        self.rag = rag
        self.uses_rag = prompt_style in rag_prompt_styles

        # Methods that generate code to be executed
        code_execution_styles = {"stepback_calc_rag", "medrac_rag"}
        self.uses_code_execution = prompt_style in code_execution_styles

        # Store latest run artefacts (optional, mainly for  backward compatibility)
        self.responses: Dict[str, List[str]] = {}
        self.answers: Dict[str, List[str]] = {}
        self.correctness: Dict[str, Dict[str, List[bool]]] = {}
        self.input_tokens: Dict[str, List[int]] = {}
        self.output_tokens: Dict[str, List[int]] = {}

        super().__init__(llms=llms, **kwargs)

    # ---------------------------------------------------------------------------------- #
    # Stage 1 – generation
    # ---------------------------------------------------------------------------------- #
    def generate_raw(
        self,
        test: bool = False,
        raw_json_dir: str = "raw_output",
    ) -> str:
        """Run the LLM(s) once, persist answers + token stats.

        Returns
        -------
        str
            Path of the raw JSON file (single-model use-case).  If multiple
            models are attached, the function still runs them all but returns
            the **first** path for convenience; a map is printed to the log.
        """

        # 1) load dataset ----------------------------------------------------------------
        if test:
            self.df = self.load_data_test()
        else:
            self.df = self.load_dataset()

        # 2) build prompts ---------------------------------------------------------------
        notes = self.df["Patient Note"].tolist()
        questions = self.df["Question"].tolist()
        calids = self.df["Calculator ID"].astype(str).tolist()

        # Call prompt function with RAG if needed
        if self.uses_rag:
            prompts = self.prompt_fn(calids, notes, questions, self.rag)
        else:
            prompts = self.prompt_fn(calids, notes, questions)

        os.makedirs(raw_json_dir, exist_ok=True)
        prompt_list = [
            {"system_msg": system, "user_msg": user}
            for system, user in prompts
        ]
        prompt_path = os.path.join(raw_json_dir, "prompts.json")
        with open(prompt_path, "w", encoding="utf-8") as f:
            json.dump(prompt_list, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(prompt_list)} prompts to {prompt_path}")
        # 3) ensure output dir -----------------------------------------------------------
        Path(raw_json_dir).mkdir(parents=True, exist_ok=True)

        # 4) query every model -----------------------------------------------------------
        model_to_path = {}
        for llm in self.llm_list:
            model_name = llm.model_name_full
            safe_model_name = model_name.replace("/", "_")  # file‑system safe

            schema = prompt_style_to_schema(self.prompt_style)
            task_desc = f"[{self.prompt_style.upper()}] Generating answers"
            generations = llm.generate(prompts, schema=schema, task_desc=task_desc)
            (
                self.responses[model_name],
                self.input_tokens[model_name],
                self.output_tokens[model_name],
            ) = map(list, zip(*generations))

            # Execute code for MedRaC-style methods
            if self.uses_code_execution:
                logger.info("Executing generated code for %d responses...", len(self.responses[model_name]))
                executed_responses = []
                for resp in self.responses[model_name]:
                    if isinstance(resp, dict):
                        code = resp.get("python_code", "")
                        computed_answer = execute_code(code)
                        # Update response with computed answer
                        resp["computed_answer"] = computed_answer
                        resp["answer"] = computed_answer
                        executed_responses.append(resp)
                    elif isinstance(resp, str):
                        try:
                            resp_dict = json.loads(resp)
                            code = resp_dict.get("python_code", "")
                            computed_answer = execute_code(code)
                            resp_dict["computed_answer"] = computed_answer
                            resp_dict["answer"] = computed_answer
                            executed_responses.append(resp_dict)
                        except json.JSONDecodeError:
                            # If not JSON, try to execute as code directly
                            computed_answer = execute_code(resp)
                            executed_responses.append({"answer": computed_answer, "raw": resp})
                    else:
                        executed_responses.append(resp)
                self.responses[model_name] = executed_responses
                logger.info("Code execution complete")

            raw_records = self._build_records(
                model_name=model_name,
                include_evaluation=False,  # only raw fields
            )

            raw_path = (
                Path(raw_json_dir)
                / f"{safe_model_name}_{self.prompt_style}_raw.json"
            )
            self._dump_json(raw_records, raw_path)
            model_to_path[model_name] = str(raw_path)

            logger.info("Raw generation complete - %s saved (%d rows)",
                        raw_path, len(raw_records))

        # -------------------------------------------------------------------------------
        # For the common 1‑model scenario return its path so that calling‑site
        # code can remain simple.
        # -------------------------------------------------------------------------------
        if len(model_to_path) == 1:
            return next(iter(model_to_path.values()))
        return json.dumps(model_to_path, indent=2)

    # ---------------------------------------------------------------------------------- #
    # Stage 2 – evaluation
    # ---------------------------------------------------------------------------------- #
    def evaluate(
        self,
        raw_json_file: str,
        eval_json_dir: str = "eval_output",
    ) -> str:
        """Read *raw* file, run evaluators, persist extended results."""
        str_raw_json_file = raw_json_file
        raw_json_file = Path(raw_json_file)
        if not raw_json_file.exists():
            raise FileNotFoundError(raw_json_file)

        # 1) read raw data
        with raw_json_file.open("r", encoding="utf‑8") as f:
            records = json.load(f)
        self.df = pd.DataFrame(records)

        # 2) infer model + bookkeeping 
        model_name = self.df["Model Name"].iloc[0]
        safe_model_name = model_name.replace("/", "_")

        self.responses[model_name] = self.df["LLM Original Answer"].tolist()
        self.input_tokens[model_name] = self.df["Input Tokens"].tolist()
        self.output_tokens[model_name] = self.df["Output Tokens"].tolist()
        self.correctness[model_name] = {}

        # 3) common fields for evaluators 
        ground_truths = self.df["Ground Truth Answer"].tolist()
        calids = self.df["Calculator ID"].astype(str).tolist()
        upper_limits = self.df["Upper Limit"].tolist()
        lower_limits = self.df["Lower Limit"].tolist()

        # 4) run every evaluator
        for evaluator in self.evaluators:
            # if (
            #     "LLM_Evaluator" in evaluator.get_evaluator_name()
            #     and self.prompt_style == "direct"
            # ):
            #     logger.debug("Skipping %s for direct style",
            #                  evaluator.get_evaluator_name())
            #     continue

            key, result_list = evaluator.check_correctness(
                responses=self.responses[model_name],
                ground_truths=ground_truths,
                calids=calids,
                upper_limits=upper_limits,
                lower_limits=lower_limits,
                relevant_entities=self.df["Relevant Entities"].tolist(),
                ground_truth_explanations=self.df["Ground Truth Explanation"].tolist(),
            )
            self.correctness[model_name][key] = result_list
            self.df[key] = result_list  # append column
            logger.info("Evaluator %s completed", evaluator.get_evaluator_name())

        # 5) token stats for LLM‑based evaluators 
        for evaluator in self.evaluators:
            if "LLM_Evaluator" in evaluator.get_evaluator_name():
                n_rows = len(self.df)
                logger.info(
                    "### LLM Evaluation Average Tokens - %s - input: %.2f | output: %.2f",
                    self.prompt_style,
                    evaluator.input_token_used / n_rows if n_rows else 0,
                    evaluator.output_token_used / n_rows if n_rows else 0,
                )

        # 6) persist full evaluation
        Path(eval_json_dir).mkdir(parents=True, exist_ok=True)
        eval_path = Path(eval_json_dir) / f"{safe_model_name}_{self.prompt_style}_eval.json"
        self._dump_json(self.df.to_dict(orient="records"), eval_path)
        logger.info("Evaluation written to %s", eval_path)

        return str(eval_path)

