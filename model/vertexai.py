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
import google.auth
from google.oauth2 import service_account
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig

nest_asyncio.apply()

# Path to credential file
CREDENTIAL_PATH = os.path.join(os.path.dirname(__file__), "vertex-ai-credential.json")

class APIModel(LLM):
    def __init__(
        self,
        model_name: str = "VertexAI/gemini-2.5-flash",
        temperature: float = 1.0,
        rpm_limit: Optional[int] = None,
        tpm_limit: Optional[int] = None,
    ):
        """
        Args:
            model_name: Model name, e.g., "VertexAI/gemini-2.5-flash"
            temperature: sampling temperature
            rpm_limit: max requests per rolling 60-second window
            tpm_limit: max tokens (in+out) per rolling 60-second window
        """
        super().__init__(model_name=model_name, temperature=temperature)

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

        # Initialize Vertex AI with service account credentials
        credentials = service_account.Credentials.from_service_account_file(
            CREDENTIAL_PATH,
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )

        # Read project_id from credential file
        with open(CREDENTIAL_PATH, 'r') as f:
            cred_data = json.load(f)
            project_id = cred_data.get("project_id")

        vertexai.init(
            project=project_id,
            location="us-central1",
            credentials=credentials
        )

        self.client = GenerativeModel(self.model_name)

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
        schema: Optional[Type[BaseModel]] = None
    ) -> Tuple[object, int, int]:
        est_tokens = (self.tokens_used / self.requests_made) * 1.5 if self.requests_made != 0 else self.max_tokens
        reserve_entry = await self._throttle(est_tokens)

        generation_config = GenerationConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
        )

        if schema is not None:
            generation_config = GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
                response_mime_type="application/json",
            )

        combined_prompt = f"{system_msg}\n\n{user_msg}"

        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.client.generate_content(
                        combined_prompt,
                        generation_config=generation_config,
                    )
                )
                break
            except Exception as e:
                print(f"Vertex AI request failed (try {attempt}/{max_retries}): {type(e).__name__}: {e}")
                if attempt < max_retries:
                    await asyncio.sleep(1)
        else:
            raise RuntimeError("Vertex AI failed after 3 retries")

        input_tokens = response.usage_metadata.prompt_token_count
        output_tokens = response.usage_metadata.candidates_token_count

        async with self._throttle_lock:
            total = input_tokens + output_tokens
            self.tokens_used += total
            self.requests_made += 1
            reserve_entry[1] = total

        text = response.text
        try:
            result = json.loads(text)
        except json.JSONDecodeError:
            result = text

        return result, input_tokens, output_tokens

    async def generate_async(
        self,
        prompts: List[Tuple[str, str]],
        schema: Optional[Type[BaseModel]] = None,
        show_progress: bool = True
    ) -> List[Tuple[str, int, int]]:

        async def _wrap(idx: int, sys_msg: str, user_msg: str):
            out = await self._generate_single(sys_msg, user_msg, schema)
            return idx, out

        tasks = [
            asyncio.create_task(_wrap(i, sys, usr))
            for i, (sys, usr) in enumerate(prompts)
        ]

        if not show_progress:
            done = await asyncio.gather(*tasks)
            done.sort(key=lambda x: x[0])
            return [res for _, res in done]

        results: List[Optional[Tuple[str, int, int]]] = [None] * len(prompts)
        for coro in tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc=f"Generating From {self.model_name}..."
        ):
            idx, res = await coro
            results[idx] = res

        return results

    def generate(
        self,
        prompts: List[Tuple[str, str]],
        schema: Optional[Type[BaseModel]] = None,
        show_progress: bool = True,
        **kwargs
    ) -> List[Tuple[str, int, int]]:
        return asyncio.run(
            self.generate_async(prompts, schema, show_progress=show_progress)
        )
