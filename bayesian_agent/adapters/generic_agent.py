"""Optional GenericAgent task-execution adapter.

GenericAgent is used here as an execution harness only. Benchmark loops and
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
DEFAULT_ANTHROPIC_BASE_URL = "https://api.deepseek.com/anthropic"


class UsageCollector:
    """Collect token usage from GenericAgent's llmcore usage hook."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_tokens = 0
        self.events: List[Mapping[str, Any]] = []

    def record(self, usage: Mapping[str, Any], api_mode: str) -> None:
        if not usage:
            return
        inp = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
        out = usage.get("completion_tokens") or usage.get("output_tokens") or 0
        total = usage.get("total_tokens") or (int(inp or 0) + int(out or 0))
        self.input_tokens += int(inp or 0)
        self.output_tokens += int(out or 0)
        self.total_tokens += int(total or 0)
        self.events.append({"api_mode": api_mode, "usage": dict(usage)})


@dataclass
class GenericAgentAdapter:
    """Run one prompt in one workspace using a local GenericAgent checkout."""

    root: Optional[str] = None
    model: str = DEFAULT_MODEL
    api_key_env: str = "DEEPSEEK_API_KEY"
    base_url: str = DEFAULT_BASE_URL
    anthropic_base_url: str = DEFAULT_ANTHROPIC_BASE_URL
    protocol: str = "openai"
    max_tokens: int = 8192
    context_win: int = 50000

    def integration_note(self) -> str:
        return (
            "GenericAgent integration is optional. GenericAgent executes task prompts; "
            "Bayesian-Agent owns benchmark orchestration and Bayesian Skill evolution. "
            "GenericAgent code is not copied or vendored."
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
        modules = self._load_genericagent_modules()
        self._install_model_config(modules["llmcore"], modules["agentmain"])
        usage = UsageCollector()
        base_record_usage = self._install_usage_hook(modules["llmcore"], usage)
        workspace_path = Path(workspace).resolve()
        workspace_path.mkdir(parents=True, exist_ok=True)

        try:
            agent = modules["agentmain"].GenericAgent()
            agent.next_llm(0)
            agent.llmclient.backend.history = []
            agent.llmclient.last_tools = ""
            agent.history = []
            agent.task_dir = str(workspace_path)
            agent.peer_hint = False
            agent.verbose = False
            handler = modules["ga"].GenericAgentHandler(agent, agent.history, str(workspace_path))
            agent.handler = handler
            agent.llmclient.log_path = str(workspace_path / "model_response_log.txt")

            usage.reset()
            started = time.time()
            gen = modules["agent_loop"].agent_runner_loop(
                agent.llmclient,
                modules["agentmain"].get_system_prompt(),
                prompt,
                handler,
                modules["agentmain"].TOOLS_SCHEMA,
                max_turns=max_turns,
                verbose=False,
            )
            transcript, exit_reason = _run_generator(gen)
            elapsed = time.time() - started
            (workspace_path / "transcript.txt").write_text(transcript, encoding="utf-8")
            return {
                "transcript": transcript,
                "exit_reason": exit_reason,
                "elapsed_seconds": elapsed,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "total_tokens": usage.total_tokens,
                "usage_events": usage.events,
            }
        finally:
            modules["llmcore"]._record_usage = base_record_usage

    def model_configs(self, api_key: str) -> MutableMapping[str, Mapping[str, Any]]:
        common = {
            "apikey": api_key,
            "model": self.model,
            "max_retries": 3,
            "connect_timeout": 10,
            "read_timeout": 180,
            "max_tokens": self.max_tokens,
            "context_win": self.context_win,
        }
        if self.protocol == "anthropic":
            return {
                "native_claude_config_bayesian_agent": {
                    **common,
                    "name": f"{self.model}-native-claude",
                    "apibase": self.anthropic_base_url,
                }
            }
        return {
            "native_oai_config_bayesian_agent": {
                **common,
                "name": f"{self.model}-native-oai",
                "apibase": self.base_url,
                "api_mode": "chat_completions",
            }
        }

    def resolve_root(self) -> Path:
        candidates: List[Path] = []
        if self.root:
            candidates.append(Path(self.root).expanduser())
        if os.environ.get("GENERICAGENT_ROOT"):
            candidates.append(Path(os.environ["GENERICAGENT_ROOT"]).expanduser())
        cwd = Path.cwd()
        candidates.extend([cwd, cwd.parent / "GenericAgent", Path(__file__).resolve().parents[3] / "GenericAgent"])
        metadata_root = _editable_root_from_metadata("genericagent")
        if metadata_root:
            candidates.append(metadata_root)
        spec = importlib.util.find_spec("agentmain")
        if spec and spec.origin:
            candidates.append(Path(spec.origin).resolve().parent)

        for candidate in candidates:
            root = candidate.resolve()
            if (root / "agentmain.py").exists() and (root / "agent_loop.py").exists():
                return root
        searched = ", ".join(str(p) for p in candidates)
        raise FileNotFoundError(f"Could not find GenericAgent checkout. Searched: {searched}")

    def _load_genericagent_modules(self) -> Mapping[str, Any]:
        root = self.resolve_root()
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        modules = {}
        for name in ("llmcore", "agent_loop", "agentmain", "ga"):
            if name in sys.modules:
                modules[name] = sys.modules[name]
            else:
                modules[name] = __import__(name)
        return modules

    def _install_model_config(self, llmcore: Any, agentmain: Any) -> None:
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"Set {self.api_key_env} before running GenericAgent tasks.")
        configs = self.model_configs(api_key)

        def reload_mykeys():
            llmcore.mykeys = configs
            return configs, True

        llmcore.reload_mykeys = reload_mykeys
        llmcore.mykeys = configs
        agentmain.reload_mykeys = reload_mykeys

    @staticmethod
    def _install_usage_hook(llmcore: Any, usage: UsageCollector):
        base = getattr(llmcore, "_bayesian_agent_base_record_usage", None)
        if base is None:
            base = llmcore._record_usage
            llmcore._bayesian_agent_base_record_usage = base

        def hooked(raw_usage, api_mode):
            usage.record(raw_usage, api_mode)
            return base(raw_usage, api_mode)

        llmcore._record_usage = hooked
        return base


def _run_generator(gen):
    chunks = []
    try:
        while True:
            chunks.append(next(gen))
    except StopIteration as exc:
        return "".join(chunks), exc.value


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
