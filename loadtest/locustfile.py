import os

from locust import HttpUser, between, task


class LLMUser(HttpUser):
    wait_time = between(0.5, 2)

    @task
    def chat(self):
        with self.client.post(
            "/v1/chat/completions",
            headers={"authorization": f"Bearer {os.environ['LLM_API_KEY']}"},
            json={
                "model": os.getenv("LLM_MODEL", "Qwen/Qwen3-8B-AWQ"),
                "messages": [{"role": "user", "content": "What does an LLM gateway do?"}],
                "max_tokens": 128,
                "temperature": 0,
                "stream": False,
            },
            catch_response=True,
            timeout=300,
        ) as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}: {response.text[:200]}")

