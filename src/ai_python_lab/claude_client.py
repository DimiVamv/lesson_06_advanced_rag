import os
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()


class ClaudeClient(Anthropic):
    def __init__(self):
        self.api_key = os.getenv('ANTHROPIC_API_KEY')

        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables.")
        elif not self.api_key.startswith("sk-"):
            raise ValueError("ANTHROPIC_API_KEY does not appear to be valid.")
        elif self.api_key.strip() != self.api_key:
            raise ValueError("ANTHROPIC_API_KEY contains leading or trailing whitespace.")

        # Initialize the Anthropic parent client so all SDK methods are inherited.
        super().__init__(api_key=self.api_key)
        # Backward compatibility for code that expects a `.client` attribute.
        self.client = self

    def ask_claude(self, prompt, max_tokens=20):
        response = self.messages.create(
            # model="claude-2",
            model="claude-haiku-4-5-20251001",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens
        )
        return response.content[0].text

    # Backward-compatible alias used by lesson/example files.
    def ask(self, prompt, max_tokens=1000):
        return self.ask_claude(prompt, max_tokens=max_tokens)

    def ask_with_system(self, system_prompt, user_prompt, max_tokens=200):
        response = self.messages.create(
            model="claude-haiku-4-5-20251001",
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=max_tokens,
        )
        return response.content[0].text

    # def ask_claude(self, prompt):
    #     try:
    #         response = self.client.completions.create(
    #             model="claude-2",
    #             prompt=prompt,
    #             max_tokens_to_sample=1000,
    #             temperature=0.7,
    #         )
    #         return response.completion
    #     except Exception as e:
    #         print(f"Error communicating with Claude: {e}")
    #         return None