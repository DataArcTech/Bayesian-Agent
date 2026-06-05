"""Optional mini-swe-agent task-execution adapter.

mini-swe-agent is used here as an execution harness only. Benchmark loops and
Bayesian Skill evolution live in Bayesian-Agent.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Mapping, MutableMapping, Optional, Union


DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_BASE_URL = "https://api.deepseek.com"


@dataclass
class MiniSWEAgentAdapter:
    """Run one prompt in one workspace using a local mini-swe-agent checkout."""

    root: Optional[str] = None
    model: str = DEFAULT_MODEL
    api_key_env: str = "DEEPSEEK_API_KEY"
    base_url: str = DEFAULT_BASE_URL
    config: str = "default"
    environment_timeout: int = 60
    wall_time_limit_seconds: int = 0

    def integration_note(self) -> str:
        return (
            "mini-swe-agent integration is optional. mini-swe-agent executes task prompts; "
            "Bayesian-Agent owns benchmark orchestration and Bayesian Skill evolution. "
            "mini-swe-agent code is not copied or vendored."
        )

    def run(self, task: Mapping[str, Any], skill_context: str = "") -> Mapping[str, Any]:
        prompt = str(task["prompt"])
        if skill_context:
            prompt = f"{skill_context}\n{prompt}"
        return self.run_task(
            prompt=prompt,
            workspace=task["workspace"],
            max_turns=int(task.get("max_turns", 8) or 8),
        )

    def build_task(self, *, prompt: str, workspace: Union[os.PathLike, str], max_turns: int = 8) -> Mapping[str, Any]:
        return {"prompt": prompt, "workspace": str(Path(workspace).resolve()), "max_turns": int(max_turns)}

    def run_task(self, *, prompt: str, workspace: Union[os.PathLike, str], max_turns: int = 8) -> Mapping[str, Any]:
        modules = self._load_minisweagent_modules()
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"Set {self.api_key_env} before running mini-swe-agent tasks.")

        workspace_path = Path(workspace).resolve()
        workspace_path.mkdir(parents=True, exist_ok=True)
        trajectory_path = workspace_path / "mini_swe_agent_trajectory.json"
        config = self._build_runtime_config(api_key, workspace_path, trajectory_path, max_turns)

        started = time.time()
        agent = modules["DefaultAgent"](
            modules["LitellmModel"](**config["model"]),
            modules["LocalEnvironment"](**config["environment"]),
            **config["agent"],
        )
        result = agent.run(prompt)
        elapsed = time.time() - started
        trajectory = agent.save(trajectory_path)
        transcript = format_transcript(trajectory.get("messages") or [])
        (workspace_path / "transcript.txt").write_text(transcript, encoding="utf-8")
        (workspace_path / "model_response_log.txt").write_text(
            json.dumps(trajectory, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        usage = collect_usage(trajectory.get("messages") or [])
        return {
            "transcript": transcript,
            "exit_reason": str(result.get("exit_status") or ""),
            "elapsed_seconds": elapsed,
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
            "total_tokens": usage["total_tokens"],
            "total_cost_usd": float(agent.cost or 0.0),
            "usage_events": usage["usage_events"],
            "api_calls": int(agent.n_calls or 0),
            "mini_swe_agent_trajectory": str(trajectory_path),
            "submission": str(result.get("submission") or ""),
        }

    def model_config(self, api_key: str) -> MutableMapping[str, Any]:
        return {
            "model_name": self.litellm_model_name(),
            "model_kwargs": {
                "api_key": api_key,
                "api_base": self.base_url,
                "drop_params": True,
            },
            "cost_tracking": "ignore_errors",
        }

    def litellm_model_name(self) -> str:
        if "/" in self.model:
            return self.model
        return f"openai/{self.model}"

    def resolve_root(self) -> Path:
        candidates: List[Path] = []
        if self.root:
            candidates.append(Path(self.root).expanduser())
        for env_name in ("MINI_SWE_AGENT_ROOT", "MINISWEAGENT_ROOT"):
            if os.environ.get(env_name):
                candidates.append(Path(os.environ[env_name]).expanduser())
        cwd = Path.cwd()
        candidates.extend([cwd, cwd / "mini-swe-agent", cwd.parent / "mini-swe-agent"])
        candidates.append(Path(__file__).resolve().parents[3] / "mini-swe-agent")
        metadata_root = _editable_root_from_metadata("mini-swe-agent")
        if metadata_root:
            candidates.append(metadata_root)
        spec = importlib.util.find_spec("minisweagent")
        if spec and spec.origin:
            candidates.append(Path(spec.origin).resolve().parents[1])

        for candidate in candidates:
            root = candidate.resolve()
            if (root / "src" / "minisweagent").exists():
                return root
            if (root / "minisweagent").exists():
                return root
        searched = ", ".join(str(p) for p in candidates)
        raise FileNotFoundError(f"Could not find mini-swe-agent checkout. Searched: {searched}")

    def _build_runtime_config(
        self,
        api_key: str,
        workspace: Path,
        trajectory_path: Path,
        max_turns: int,
    ) -> MutableMapping[str, Any]:
        config = self._load_config()
        agent_config = dict(config.get("agent") or {})
        model_config = dict(config.get("model") or {})
        environment_config = dict(config.get("environment") or {})
        model_config = _deep_merge(model_config, self.model_config(api_key))
        environment_config = _deep_merge(
            environment_config,
            {"cwd": str(workspace), "timeout": int(self.environment_timeout)},
        )
        agent_config["step_limit"] = int(max_turns or 0)
        agent_config["output_path"] = trajectory_path
        agent_config["wall_time_limit_seconds"] = int(self.wall_time_limit_seconds or 0)
        return {"agent": agent_config, "model": model_config, "environment": environment_config}

    def _load_config(self) -> MutableMapping[str, Any]:
        self._ensure_minisweagent_importable()
        from minisweagent.config import get_config_from_spec

        return dict(get_config_from_spec(self.config))

    def _load_minisweagent_modules(self) -> Mapping[str, Any]:
        self._ensure_minisweagent_importable()
        from minisweagent.agents.default import DefaultAgent
        from minisweagent.environments.local import LocalEnvironment
        from minisweagent.models.litellm_model import LitellmModel

        return {
            "DefaultAgent": DefaultAgent,
            "LocalEnvironment": LocalEnvironment,
            "LitellmModel": LitellmModel,
        }

    def _ensure_minisweagent_importable(self) -> None:
        os.environ.setdefault("MSWEA_SILENT_STARTUP", "1")
        root = self.resolve_root()
        source_root = root / "src"
        for path in (source_root, root):
            if path.exists() and str(path) not in sys.path:
                sys.path.insert(0, str(path))


def format_transcript(messages: List[Mapping[str, Any]]) -> str:
    lines = []
    for idx, message in enumerate(messages, 1):
        role = str(message.get("role") or "message")
        content = message.get("content") or ""
        lines.append(f"\n\nTurn {idx} [{role}]\n{content}")
        extra = dict(message.get("extra") or {})
        actions = extra.get("actions") or []
        if actions:
            lines.append("actions=" + json.dumps(actions, ensure_ascii=False, default=str))
    return "\n".join(lines).strip()


def collect_usage(messages: List[Mapping[str, Any]]) -> Mapping[str, Any]:
    input_tokens = 0
    output_tokens = 0
    total_tokens = 0
    events = []
    for message in messages:
        response = dict((message.get("extra") or {}).get("response") or {})
        usage = dict(response.get("usage") or {})
        if not usage:
            continue
        inp = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        out = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
        total = int(usage.get("total_tokens") or inp + out)
        input_tokens += inp
        output_tokens += out
        total_tokens += total
        events.append({"source": "mini_swe_agent", "usage": usage})
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "usage_events": events,
    }


def _deep_merge(base: MutableMapping[str, Any], overlay: Mapping[str, Any]) -> MutableMapping[str, Any]:
    result = dict(base)
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(dict(result[key]), value)
        else:
            result[key] = value
    return result


def _editable_root_from_metadata(distribution_name: str) -> Optional[Path]:
    try:
        dist = importlib.metadata.distribution(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return None
    direct_url = dist.read_text("direct_url.json")
    if not direct_url:
        return None
    try:
        data = json.loads(direct_url)
    except json.JSONDecodeError:
        return None
    url = str(data.get("url") or "")
    if not url.startswith("file://"):
        return None
    return Path(url[7:]).expanduser()
