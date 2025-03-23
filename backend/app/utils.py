from contextlib import contextmanager
import time

@contextmanager
def track_token_usage():
    class TokenUsage:
        def __init__(self):
            self.prompt_tokens = 0
            self.completion_tokens = 0
            self.total_tokens = 0
            self.start_time = time.time()
            self.end_time = None
            self.cost = 0

        def update(self, response):
            usage = response.get("usage", {})
            self.prompt_tokens += usage.get("prompt_tokens", 0)
            self.completion_tokens += usage.get("completion_tokens", 0)
            self.total_tokens += usage.get("total_tokens", 0)
            self.cost = (self.prompt_tokens / 1000) * 0.003 + (self.completion_tokens / 1000) * 0.004

        def finish(self):
            self.end_time = time.time()

        def __str__(self):
            duration = round(self.end_time - self.start_time, 2) if self.end_time else 0
            return f"Tokens: {self.total_tokens}, Cost: ${self.cost:.6f}, Duration: {duration}s"

    token_usage = TokenUsage()
    try:
        yield token_usage
    finally:
        token_usage.finish()
        print(token_usage)