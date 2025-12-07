import os
import json
import asyncio
import time
from typing import List, Tuple, Optional, Type
from collections import deque
from pydantic import BaseModel
from model.model import LLM
from tqdm import tqdm
import nest_asyncio
from google.oauth2 import service_account

# New unified SDK (pip install google-genai)
from google import genai
from google.genai import types

nest_asyncio.apply()

# Path to credential file
CREDENTIAL_PATH = os.path.join(os.path.dirname(__file__), "vertex-ai-credential.json")

class APIModel(LLM):
    def __init__(
        self,
        model_name: str = "VertexAI/gemini-2.0-flash",
        temperature: float = 1.0,
        rpm_limit: Optional[int] = None,
        tpm_limit: Optional[int] = None,
        disable_thinking: bool = True,
        max_concurrency: Optional[int] = None,
    ):
        """
        Args:
            model_name: Model name, e.g., "VertexAI/gemini-2.5-flash"
            temperature: sampling temperature
            rpm_limit: max requests per rolling 60-second window
            tpm_limit: max tokens (in+out) per rolling 60-second window
            disable_thinking: If True, disable thinking mode for 2.5 models (default: True)
            max_concurrency: Max concurrent requests. None=sequential, 10-20=recommended parallel
        """
        self.max_concurrency = max_concurrency
        super().__init__(model_name=model_name, temperature=temperature)

        self.disable_thinking = disable_thinking

        # rate-limiting state
        self._req_times = deque()
        self._token_times = deque()
        self._throttle_lock = asyncio.Lock()

        max_tokens_map = {
            "gemini-2.5-flash": 65536,
            "gemini-2.5-pro": 65536,
            "gemini-2.0-flash": 8192,
            "gemini-1.5-flash": 8192,
            "gemini-1.5-pro": 8192,
        }

        rpm_map = {
            "gemini-2.5-flash": 1000,
            "gemini-2.5-pro": 100,
            "gemini-2.0-flash": 1000,
            "gemini-1.5-flash": 1000,
            "gemini-1.5-pro": 100,
        }

        tpm_map = {
            "gemini-2.5-flash": 4000000,
            "gemini-2.5-pro": 2000000,
            "gemini-2.0-flash": 4000000,
            "gemini-1.5-flash": 4000000,
            "gemini-1.5-pro": 2000000,
        }

        self.max_tokens = max_tokens_map.get(self.model_name, 8192)
        self.rpm_limit = rpm_limit or rpm_map.get(self.model_name, 500)
        self.tpm_limit = tpm_limit or tpm_map.get(self.model_name, 2000000)
        self.tokens_used = 0
        self.requests_made = 0
        print(f'{self.model_name} max_tokens: {self.max_tokens}, rpm_limit: {self.rpm_limit}, tpm_limit: {self.tpm_limit}')

        # Read project_id from credential file
        with open(CREDENTIAL_PATH, 'r') as f:
            cred_data = json.load(f)
            self.project_id = cred_data.get("project_id")

        # Set credentials for google-genai SDK
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = CREDENTIAL_PATH

        # Initialize new unified SDK client (pip install google-genai)
        self.client = genai.Client(
            vertexai=True,
            project=self.project_id,
            location="us-central1"
        )

        if self.disable_thinking and "2.5" in self.model_name:
            print(f"  Thinking mode: DISABLED (thinking_budget=1)")

        if self.max_concurrency is not None:
            print(f"  Concurrency: PARALLEL (max {self.max_concurrency} concurrent requests)")
        else:
            print(f"  Concurrency: SEQUENTIAL")

    async def _throttle(self, tokens_est: int):
        async with self._throttle_lock:
            now = time.monotonic()

            while self._req_times and now - self._req_times[0] > 60:
                self._req_times.popleft()
            while self._token_times and now - self._token_times[0][0] > 60:
                self._token_times.popleft()

            rpm_wait = 0
            if self.rpm_limit and len(self._req_times) >= self.rpm_limit:
                oldest = self._req_times[0]
                rpm_wait = (oldest + 60) - now

            tpm_wait = 0
            if self.tpm_limit:
                used = sum(toks for _, toks in self._token_times)
                needed = used + tokens_est - self.tpm_limit
                if needed > 0:
                    freed = 0
                    for ts, toks in self._token_times:
                        freed += toks
                        if freed >= needed:
                            tpm_wait = (ts + 60) - now
                            break
                    else:
                        last_ts = self._token_times[-1][0]
                        tpm_wait = (last_ts + 60) - now

        wait = max(rpm_wait, tpm_wait, 0)
        if wait > 0:
            await asyncio.sleep(wait)
            return await self._throttle(tokens_est)

        async with self._throttle_lock:
            self._req_times.append(now)
            reserve = [now, tokens_est]
            self._token_times.append(reserve)
            return reserve

    async def _generate_single(
        self,
        system_msg: str,
        user_msg: str,
        schema: Optional[Type[BaseModel]] = None,
        request_id: int = 0
    ) -> Tuple[object, int, int]:
        """Each request has its own independent retry loop (max 10 retries)."""
        est_tokens = (self.tokens_used / self.requests_made) * 1.5 if self.requests_made != 0 else self.max_tokens
        reserve_entry = await self._throttle(est_tokens)

        # Build generation config using new unified SDK
        config_params = {
            "temperature": self.temperature,
            "max_output_tokens": self.max_tokens,
        }

        # Minimize thinking mode for Gemini 2.5 to reduce token usage
        # Note: thinking_budget=0 is not supported, use minimum value of 1
        if self.disable_thinking and "2.5" in self.model_name:
            config_params["thinking_config"] = types.ThinkingConfig(
                include_thoughts=False,
                thinking_budget=1  # Minimum thinking budget
            )

        if schema is not None:
            config_params["response_mime_type"] = "application/json"

        generation_config = types.GenerateContentConfig(**config_params)

        combined_prompt = f"{system_msg}\n\n{user_msg}"

        # Each request has independent retry counter (resets to 0 for each new request)
        max_retries = 10
        for attempt in range(1, max_retries + 1):
            try:
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.client.models.generate_content(
                        model=self.model_name,
                        contents=combined_prompt,
                        config=generation_config,
                    )
                )
                break
            except Exception as e:
                error_msg = str(e)
                is_rate_limit = "429" in error_msg or "ResourceExhausted" in error_msg or "RESOURCE_EXHAUSTED" in error_msg

                if attempt < max_retries:
                    # Exponential backoff: 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024 seconds
                    wait_time = 2 ** attempt
                    if is_rate_limit:
                        print(f"[Req {request_id}] Rate limited (try {attempt}/{max_retries}), waiting {wait_time}s...")
                    else:
                        # Show full error message for debugging
                        print(f"[Req {request_id}] Failed (try {attempt}/{max_retries}): {type(e).__name__}: {str(e)[:200]}")
                    await asyncio.sleep(wait_time)
                else:
                    print(f"[Req {request_id}] Failed (try {attempt}/{max_retries}): {type(e).__name__}: {e}")
        else:
            raise RuntimeError(f"[Req {request_id}] Vertex AI failed after 10 retries")

        # Extract token counts from new SDK response format
        # Handle None values for token counts (can happen with certain responses)
        input_tokens = 0
        output_tokens = 0
        if response.usage_metadata:
            input_tokens = response.usage_metadata.prompt_token_count or 0
            output_tokens = response.usage_metadata.candidates_token_count or 0

        async with self._throttle_lock:
            total = input_tokens + output_tokens
            self.tokens_used += total
            self.requests_made += 1
            reserve_entry[1] = total

        # Handle cases where response has no text (e.g., MAX_TOKENS, safety filters)
        try:
            text = response.text
        except (ValueError, AttributeError) as e:
            # Check finish reason
            finish_reason = None
            if response.candidates:
                finish_reason = response.candidates[0].finish_reason
            print(f"Warning: No response text. Finish reason: {finish_reason}. Error: {e}")
            return {"error": str(finish_reason), "answer": ""}, input_tokens, output_tokens

        try:
            result = json.loads(text)
        except json.JSONDecodeError:
            result = text

        return result, input_tokens, output_tokens

    async def generate_async(
        self,
        prompts: List[Tuple[str, str]],
        schema: Optional[Type[BaseModel]] = None,
        show_progress: bool = True,
        task_desc: Optional[str] = None,
        max_concurrency: Optional[int] = None,
    ) -> List[Tuple[str, int, int]]:
        """
        Generate responses with configurable parallelism.

        Args:
            max_concurrency: Max concurrent requests. None = sequential, 0 = unlimited.
                            Recommended: 5-20 depending on rate limits.
        """
        desc = task_desc if task_desc else f"Generating ({self.model_name})"

        # Sequential mode (default for backward compatibility)
        if max_concurrency is None:
            results: List[Tuple[str, int, int]] = []
            iterator = enumerate(prompts)
            if show_progress:
                iterator = tqdm(list(iterator), desc=desc)
            for idx, (sys_msg, user_msg) in iterator:
                result = await self._generate_single(sys_msg, user_msg, schema, request_id=idx)
                results.append(result)
            return results

        # Parallel mode with semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency > 0 else None
        results = [None] * len(prompts)
        completed = [0]  # Use list for mutable counter in closure

        pbar = tqdm(total=len(prompts), desc=desc) if show_progress else None

        async def process_one(idx: int, sys_msg: str, user_msg: str):
            if semaphore:
                async with semaphore:
                    result = await self._generate_single(sys_msg, user_msg, schema, request_id=idx)
            else:
                result = await self._generate_single(sys_msg, user_msg, schema, request_id=idx)

            results[idx] = result
            completed[0] += 1
            if pbar:
                pbar.update(1)
            return result

        # Create all tasks
        tasks = [
            process_one(idx, sys_msg, user_msg)
            for idx, (sys_msg, user_msg) in enumerate(prompts)
        ]

        # Run all tasks concurrently
        await asyncio.gather(*tasks)

        if pbar:
            pbar.close()

        return results

    def generate(
        self,
        prompts: List[Tuple[str, str]],
        schema: Optional[Type[BaseModel]] = None,
        show_progress: bool = True,
        task_desc: Optional[str] = None,
        max_concurrency: Optional[int] = -1,  # -1 means use instance default
        **kwargs
    ) -> List[Tuple[str, int, int]]:
        """
        Generate responses for multiple prompts.

        Args:
            max_concurrency: Max concurrent requests.
                            -1 = use instance default (self.max_concurrency)
                            None = sequential (safest)
                            10-20 = recommended for parallel with rate limiting
                            0 = unlimited (use with caution)
        """
        # Use instance default if not specified
        if max_concurrency == -1:
            max_concurrency = self.max_concurrency

        return asyncio.run(
            self.generate_async(
                prompts, schema,
                show_progress=show_progress,
                task_desc=task_desc,
                max_concurrency=max_concurrency
            )
        )
