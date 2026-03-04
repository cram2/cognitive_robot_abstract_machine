import os
from unittest import skip

import requests
from IPython.terminal.embed import InteractiveShellEmbed
# from openai import OpenAI
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
from prompt_toolkit.key_binding import KeyBindings
from traitlets.config import Config
import pprint


class ContextAwareCopilotShell:
    def __init__(self, scope=None):
        self.scope = scope or {}
        self.shell = self._init_shell()

    def _init_shell(self):
        cfg = Config()
        shell = InteractiveShellEmbed(config=cfg, user_ns=self.scope)
        self._add_keybindings(shell)
        return shell

    def _add_keybindings(self, shell):
        kb = KeyBindings()

        @kb.add('c-space')  # Trigger with Ctrl+Space
        def suggest(event):
            buffer = event.app.current_buffer
            current_code = buffer.document.text.strip()
            context = self._get_context()

            print("\n🔍 Generating context-aware suggestion...\n")
            # suggestion = self.get_suggestion(current_code, context)
            suggestion = self.get_suggestion_local(current_code, context)
            print(f"\n💡 Suggestion:\n{suggestion}\n")

        shell.pt_app.key_bindings = kb

    def _get_context(self):
        """Create a summary of local variables for the prompt."""
        ctx = self.shell.user_ns.copy()
        summary = []

        for k, v in ctx.items():
            if k.startswith("__") or callable(v):
                continue
            try:
                typename = type(v).__name__
                valstr = pprint.pformat(v, width=40)
                summary.append(f"{k} ({typename}): {valstr}")
            except Exception:
                continue

        return "\n".join(summary[-10:])  # Limit to last 10 vars

    def get_suggestion(self, code, context_vars):
        prompt = f"""You're a Python assistant helping inside a live debugger shell.
Current variables:
{context_vars}

Based on this context, suggest how to complete or improve the following Python input:
{code}

Respond with a Python snippet that makes sense in context.
"""
        try:
            response = client.chat.completions.create(model="gpt-4",
                                                      messages=[
                                                          {"role": "system",
                                                           "content": "You're a helpful Python code assistant."},
                                                          {"role": "user", "content": prompt}
                                                      ],
                                                      max_tokens=150,
                                                      temperature=0.3)
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"❌ Error: {e}"

    def get_suggestion_local(self, code, context_vars):
        prompt = f"""You are a helpful Python assistant.
    Current variables:
    {context_vars}

    Suggest a Python code snippet to complete this input:
    {code}
    """
        print(f"Prompt:\n{prompt}\n")
        try:
            response = requests.post("http://localhost:11434/api/generate", json={
                "model": "codellama:7b-instruct",  # or "phi"
                "prompt": prompt,
                "stream": False,
            })
            result = response.json()
            return result.get("response", "").strip()
        except Exception as e:
            return f"❌ Local LLM error: {e}"

    def run(self):
        self.shell.user_ns.update(self.scope)
        self.shell()
        self._sync_back()

    def _sync_back(self):
        # Keep scope up-to-date after running
        for k in self.scope:
            self.scope[k] = self.shell.user_ns.get(k, self.scope[k])


def run_shell():
    user = {"name": "Alice", "age": 30}
    data = [3, 1, 4]
    shell = ContextAwareCopilotShell(locals())
    shell.run()


# run_shell()
