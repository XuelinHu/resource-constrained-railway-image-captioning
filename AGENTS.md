# Agent Instructions

## Default Conda Environment
- Environment name: `rc-llm-eval`
- Environment path: `/home/xuelin/miniconda3/envs/rc-llm-eval`
- Prefer running Python commands with `conda run -n rc-llm-eval ...` or `/home/xuelin/miniconda3/envs/rc-llm-eval/bin/python`.

<!-- codex-agent-runtime:start -->

## Runtime Ports And Database Configuration

- Keep this section aligned with the root README when database names, ports, or service defaults change.
- Do not copy secrets from local `.env` files into commits; document only placeholders or compose defaults.

### Database
- No application database is used. Inputs and outputs are local JSONL, image, ontology, and experiment files.

### Default Ports
- No default web service or database port is defined.

### Notes For Codex Agents
- Use local config files under `configs/` and CLI entry points from `pyproject.toml`.
- Before committing, check `git status --short --branch` and avoid staging unrelated runtime artifacts.

### Source Files Checked
- `pyproject.toml`
- `configs/default.yaml`
- `README.md`

<!-- codex-agent-runtime:end -->

## GitHub Commit Language

- Use English for all GitHub commit messages and pull/push related commit notes.
