from model import LLM
from typing import Union, Tuple, List, Dict, Optional
import json
import os
from textwrap import dedent
from tqdm import tqdm
from method import Method
from evaluator import Evaluator
import logging
import pandas as pd

logger = logging.getLogger(__name__)


class StepBack(Method):
    """
    Step-Back Prompting method for medical calculations.

    This method implements the step-back prompting technique where:
    1. First, the model abstracts the problem to identify high-level principles,
       relevant medical concepts, and the general approach needed.
    2. Then, the model uses those principles to solve the specific question.

    Reference: "Take a Step Back: Evoking Reasoning via Abstraction in Large Language Models"
    """

    def __init__(
        self,
        llms: List[LLM],
        evaluators: List[Evaluator],
        use_rag: bool = False,
    ):
        """
        Initialize the StepBack method.

        Args:
            llms: List of LLM instances to use for generation.
            evaluators: List of evaluators for checking correctness.
            use_rag: Whether to use RAG for formula retrieval.
        """
        super().__init__(llms=llms)
        self.evaluators = evaluators
        self.prompt_style = "step_back_rag" if use_rag else "step_back"

        self.input_tokens = {}
        self.output_tokens = {}
        self.responses = {}
        self.correctness = {}
        self.principles = {}  # Store abstracted principles
        self.formulas = {}

        if use_rag:
            from method.rag import RAG
            self.rag = RAG()
        else:
            self.rag = None

    def generate_raw(
        self,
        test: bool = False,
        raw_json_dir: str = "raw_output/step_back",
        formula_json_path: str = "data/formula_new.json",
    ) -> str:
        """
        Run the two-stage step-back prompting and persist answers + token stats.

        Stage 1: Abstract the problem to identify principles and concepts
        Stage 2: Apply principles to solve the specific problem
        """

        # 1) Load dataset
        self.df = self.load_data_test() if test else self.load_dataset()

        notes = self.df["Patient Note"].tolist()
        questions = self.df["Question"].tolist()
        calids = self.df["Calculator ID"].astype(str).tolist()

        # 2) Ensure output dir exists
        os.makedirs(raw_json_dir, exist_ok=True)

        model_to_path = {}

        for llm in self.llm_list:
            model_name = llm.model_name_full
            safe_model_name = model_name.replace("/", "_")

            # Get formulas for reference
            if self.rag:
                formulas = [self.rag.retrieve(question, k=1)[0][0] for question in questions]
            else:
                formulas = self._get_formulas(calids=calids, json_path=formula_json_path)
            self.formulas[model_name] = formulas

            # ============ STAGE 1: Abstraction ============
            # Generate high-level principles and concepts
            abstraction_prompts = self._gen_abstraction_prompts(notes, questions)

            logger.info("Stage 1: Generating abstractions for %s...", model_name)
            abstraction_results = llm.generate(abstraction_prompts)

            principles_list, abs_in_toks, abs_out_toks = map(list, zip(*abstraction_results))
            self.principles[model_name] = principles_list

            # ============ STAGE 2: Grounded Reasoning ============
            # Use the abstracted principles to solve the problem
            reasoning_prompts = self._gen_reasoning_prompts(
                notes, questions, principles_list, formulas
            )

            logger.info("Stage 2: Generating final answers for %s...", model_name)
            reasoning_results = llm.generate(reasoning_prompts)

            answers_list, reas_in_toks, reas_out_toks = map(list, zip(*reasoning_results))

            # Combine token counts from both stages
            self.input_tokens[model_name] = [a + b for a, b in zip(abs_in_toks, reas_in_toks)]
            self.output_tokens[model_name] = [a + b for a, b in zip(abs_out_toks, reas_out_toks)]

            # Build combined responses
            self.responses[model_name] = []
            for principle, answer in zip(principles_list, answers_list):
                combined = {
                    "abstracted_principles": principle,
                    "final_response": answer,
                }
                self.responses[model_name].append(combined)

            # Dump raw records
            raw_records = self._build_records(model_name=model_name, include_evaluation=False)
            raw_path = os.path.join(
                raw_json_dir,
                f"{safe_model_name}_{self.prompt_style}_raw.json"
            )
            self._dump_json(raw_records, raw_path)
            model_to_path[model_name] = raw_path

            logger.info("Raw generation complete – %s saved (%d rows)", raw_path, len(raw_records))

        if len(model_to_path) == 1:
            return next(iter(model_to_path.values()))
        return json.dumps(model_to_path, indent=2)

    def evaluate(
        self,
        raw_json_file: str,
        eval_json_dir: str = "eval_output/step_back",
    ) -> str:
        """Read raw file, run evaluators, persist extended results."""

        if not os.path.exists(raw_json_file):
            raise FileNotFoundError(raw_json_file)

        # 1) Read raw data
        with open(raw_json_file, "r", encoding="utf-8") as f:
            records = json.load(f)
        self.df = pd.DataFrame(records)

        # 2) Infer model + bookkeeping
        model_name = self.df["Model Name"].iloc[0]
        safe_model_name = model_name.replace("/", "_")

        self.responses[model_name] = self.df["LLM Original Answer"].tolist()
        self.input_tokens[model_name] = self.df["Input Tokens"].tolist()
        self.output_tokens[model_name] = self.df["Output Tokens"].tolist()
        self.correctness[model_name] = {}

        calids = self.df["Calculator ID"].astype(str).tolist()
        self.formulas[model_name] = self._get_formulas(
            calids=calids,
            json_path="data/formula_new.json",
        )

        # 3) Common fields for evaluators
        ground_truths = self.df["Ground Truth Answer"].tolist()
        calids = self.df["Calculator ID"].astype(str).tolist()
        upper_limits = self.df["Upper Limit"].tolist()
        lower_limits = self.df["Lower Limit"].tolist()

        # 4) Run every evaluator
        for evaluator in self.evaluators:
            key, result_list = evaluator.check_correctness(
                responses=self.responses[model_name],
                ground_truths=ground_truths,
                calids=calids,
                upper_limits=upper_limits,
                lower_limits=lower_limits,
                relevant_entities=self.df["Relevant Entities"].tolist(),
                ground_truth_explanations=self.df["Ground Truth Explanation"].tolist(),
                formulas=self.formulas[model_name],
            )
            self.correctness[model_name][key] = result_list
            self.df[key] = result_list
            logger.info("Evaluator %s completed", evaluator.get_evaluator_name())

        # 5) Token stats for LLM-based evaluators
        for evaluator in self.evaluators:
            if "LLM_Evaluator" in evaluator.get_evaluator_name():
                n_rows = len(self.df)
                logger.info(
                    "### LLM Evaluation Average Tokens - %s - input: %.2f | output: %.2f",
                    self.prompt_style,
                    evaluator.input_token_used / n_rows if n_rows else 0,
                    evaluator.output_token_used / n_rows if n_rows else 0,
                )

        # 6) Persist full evaluation
        os.makedirs(eval_json_dir, exist_ok=True)
        eval_path = os.path.join(
            eval_json_dir,
            f"{safe_model_name}_{self.prompt_style}_eval.json"
        )
        self._dump_json(self.df.to_dict(orient="records"), eval_path)
        logger.info("Evaluation written to %s", eval_path)

        return eval_path

    def _gen_abstraction_prompts(
        self,
        notes: List[str],
        questions: List[str],
    ) -> List[Tuple[str, str]]:
        """
        Generate Stage 1 prompts that ask the model to abstract the problem
        and identify high-level principles, concepts, and the general approach.
        """
        prompts = []

        system_msg = dedent("""
            You are a medical expert assistant. Before solving a specific medical
            calculation problem, you should first take a step back and think about
            the underlying principles and concepts.

            Your task is to:
            1. Identify what type of medical calculation this is (e.g., risk score,
               drug dosing, physiological measurement, severity index).
            2. List the key medical concepts and principles relevant to this calculation.
            3. Identify what formula or scoring system is typically used.
            4. Note any important clinical considerations or edge cases.

            Return your response as a JSON object:
            {
                "calculation_type": str,
                "key_concepts": [list of relevant medical concepts],
                "typical_formula": str,
                "clinical_considerations": str,
                "variables_needed": [list of variables typically required]
            }
        """).strip()

        for note, question in zip(notes, questions):
            user_msg = dedent(f"""
                Patient Note:
                {note}

                Question:
                {question}

                Take a step back and identify the high-level principles and concepts
                needed to solve this problem. What type of calculation is this? What
                medical knowledge is required? What formula should be used?
            """).strip()

            prompts.append((system_msg, user_msg))

        return prompts

    def _gen_reasoning_prompts(
        self,
        notes: List[str],
        questions: List[str],
        principles: List[Union[str, dict]],
        formulas: List[str],
    ) -> List[Tuple[str, str]]:
        """
        Generate Stage 2 prompts that use the abstracted principles
        to solve the specific problem.
        """
        prompts = []

        system_msg = dedent("""
            You are a medical calculation assistant. You have already identified the
            high-level principles and concepts for this problem. Now apply that
            understanding to solve the specific calculation.

            Follow these steps:
            1. Extract the specific values from the patient note that match the
               variables needed for the formula.
            2. Apply the formula using the extracted values.
            3. Perform the calculation step by step.
            4. Provide the final answer.

            Return your response as a JSON object:
            {
                "extracted_values": {variable: value},
                "calculation_steps": str,
                "answer": str
            }
        """).strip()

        for note, question, principle, formula in zip(notes, questions, principles, formulas):
            # Format principle for context
            if isinstance(principle, dict):
                principle_text = json.dumps(principle, indent=2)
            else:
                principle_text = str(principle)

            user_msg = dedent(f"""
                === ABSTRACTED PRINCIPLES (from step-back analysis) ===
                {principle_text}

                === REFERENCE FORMULA ===
                {formula}

                === PATIENT NOTE ===
                {note}

                === QUESTION ===
                {question}

                Using the principles identified above and the reference formula,
                extract the required values from the patient note and calculate the answer.
            """).strip()

            prompts.append((system_msg, user_msg))

        return prompts

    @staticmethod
    def _get_formulas(calids: List[str], json_path: str) -> List[str]:
        """
        Given a list of Calculator IDs, return a list of the corresponding formulas
        by looking them up in a JSON file.
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        id_to_formula = {
            item["Calculator ID"]: item["Formula"]
            for item in data
        }

        formulas = []
        for cid in calids:
            if cid not in id_to_formula:
                raise ValueError(f"Calculator ID '{cid}' not found in {json_path}")
            formulas.append(id_to_formula[cid])

        return formulas
