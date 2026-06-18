# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt
pip install -e .          # installs the local package in editable mode
cp .env.example .env      # then add ANTHROPIC_API_KEY to .env
```

## Commands

```bash
# Run the interactive CLI
python -m ai_python_lab.main

# Run an example or lesson file
python examples/example_01_simple_prompt.py
python lessons/lesson_02_prompting/prompt_examples.py

# Run all tests
pytest tests/

# Run a single test
pytest tests/test_utils.py::test_exit_command_exit
```

## Architecture

The project uses a `src/` layout. The installable package lives in `src/ai_python_lab/` and is installed via `pip install -e .`. All lesson, example, and test files import from `ai_python_lab` (not from a relative path).

**`src/ai_python_lab/claude_client.py`** — central wrapper around the Anthropic SDK. Three public methods:
- `ask_claude(prompt, max_tokens)` — plain user prompt
- `ask_with_system(system_prompt, user_prompt, max_tokens)` — adds a system prompt for role/persona control
- `ask(prompt, max_tokens)` — alias for `ask_claude`, kept for backwards compatibility

**`src/ai_python_lab/utils.py`** — small helpers: `print_title()` and `is_exit_command()`.

**`lessons/`** — each subdirectory is a self-contained lesson script. Lesson files are run directly (`python lessons/lessonXX/file.py`), not imported.

**`examples/`** — standalone runnable scripts. Most use `ClaudeClient`; `calculator_example.py` is an exception (PyQt6 GUI, no API calls) and requires `uv pip install PyQt6`.

**`mini_projects/`** — Markdown spec files only (no runnable code); used as student exercise descriptions.

**`lessons/lesson_04_testing/`** — contains its own `calculator.py` + `test_calculator.py` pair that are self-contained and do not use `ClaudeClient`.

## Model in use

`ClaudeClient` defaults to `claude-haiku-4-5-20251001`. When updating lesson or example files, keep this model unless there is a specific reason to change it.
