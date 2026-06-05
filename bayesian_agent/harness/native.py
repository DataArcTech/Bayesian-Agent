"""First-party Bayesian-Agent harness runner."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from bayesian_agent.harness.llm import DEFAULT_BASE_URL, OpenAIChatClient, normalize_usage
from bayesian_agent.harness.tools import TOOL_SCHEMAS, WorkspaceToolbox, compact_tool_result


DEFAULT_SYSTEM_PROMPT = """You are Bayesian-Agent's first-party task harness.

You solve tasks inside one workspace using only the provided tools.
- Use file_read to inspect task files.
- Use code_run for calculations, CSV edits, SQL generation checks, and deterministic scripts.
- Use file_write for final output files when direct writing is simpler.
- All tool paths are scoped to the current task workspace.
- Current task files and grader output are authoritative.
- Finish only after required files are written and verified.
"""


@dataclass
class NativeBayesianAgentAdapter:
    """Run tasks with Bayesian-Agent's own LLM/tool loop, not an external harness."""

    model: str = "deepseek-v4-flash"
    api_key_env: str = "DEEPSEEK_API_KEY"
    base_url: str = DEFAULT_BASE_URL
    timeout_seconds: int = 180
    max_tokens: int = 4096
    temperature: float = 0.0
    verify_ssl: bool = True
    host_header: str = ""
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    client: Optional[Any] = None

    def integration_note(self) -> str:
        return (
            "Bayesian-Agent first-party harness executes the LLM loop, tool dispatch, "
            "workspace I/O, and trajectory capture without delegating to GenericAgent, "
            "mini-swe-agent, or Claude Code."
        )

    def run(self, task: Mapping[str, Any], skill_context: str = "") -> Mapping[str, Any]:
        prompt = str(task["prompt"])
        if skill_context:
            prompt = f"{skill_context}\n\n{prompt}"
        return self.run_task(
            prompt=prompt,
            workspace=task["workspace"],
            max_turns=int(task.get("max_turns", 8) or 8),
        )

    def build_task(self, *, prompt: str, workspace: Union[str, Path], max_turns: int = 8) -> Mapping[str, Any]:
        return {"prompt": prompt, "workspace": str(Path(workspace).resolve()), "max_turns": int(max_turns)}

    def resolve_root(self) -> Path:
        return Path(__file__).resolve().parents[2]

    def run_task(self, *, prompt: str, workspace: Union[str, Path], max_turns: int = 8) -> Mapping[str, Any]:
        workspace_path = Path(workspace).resolve()
        workspace_path.mkdir(parents=True, exist_ok=True)
        toolbox = WorkspaceToolbox(workspace_path, default_timeout=self.timeout_seconds)
        client = self._client()
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        transcript_parts: List[str] = []
        trajectory: List[Mapping[str, Any]] = []
        totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        exit_reason = "max_turns"
        started = time.time()

        for turn in range(1, int(max_turns or 8) + 1):
            response = client.chat(messages, TOOL_SCHEMAS)
            usage = normalize_usage(response.get("usage") or {})
            _accumulate_usage(totals, usage)
            content = str(response.get("content") or "")
            tool_calls = list(response.get("tool_calls") or [])
            finish_reason = str(response.get("finish_reason") or "")
            transcript_parts.append(f"\n\nTurn {turn} assistant\n{content}".rstrip())
            assistant_message = _assistant_message(content, tool_calls)
            messages.append(assistant_message)
            turn_record: Dict[str, Any] = {
                "turn": turn,
                "assistant": {"content": content, "tool_calls": tool_calls, "finish_reason": finish_reason},
                "usage": usage,
                "tool_results": [],
            }

            if not tool_calls:
                exit_reason = finish_reason or "stop"
                trajectory.append(turn_record)
                break

            finished = False
            for index, call in enumerate(tool_calls):
                tool_name = str(call.get("name") or "")
                arguments = dict(call.get("arguments") or {})
                call_id = str(call.get("id") or f"call_{turn}_{index}")
                result = toolbox.dispatch(tool_name, arguments)
                transcript_parts.append(
                    "\n".join(
                        [
                            f"Tool {tool_name}",
                            f"args={json.dumps(arguments, ensure_ascii=False, default=str)}",
                            f"result={json.dumps(result, ensure_ascii=False, default=str)}",
                        ]
                    )
                )
                messages.append({"role": "tool", "tool_call_id": call_id, "content": compact_tool_result(result)})
                turn_record["tool_results"].append({"id": call_id, "name": tool_name, "arguments": arguments, "result": result})
                if tool_name == "finish" and result.get("finished"):
                    exit_reason = "finish_tool"
                    finished = True
            trajectory.append(turn_record)
            if finished:
                break

        elapsed = time.time() - started
        transcript = "\n".join(part for part in transcript_parts if part).strip()
        payload = {
            "transcript": transcript,
            "exit_reason": exit_reason,
            "elapsed_seconds": elapsed,
            "input_tokens": totals["input_tokens"],
            "output_tokens": totals["output_tokens"],
            "total_tokens": totals["total_tokens"],
            "usage_events": [{"source": "bayesian_agent_native", "usage": totals}],
            "api_calls": len([item for item in trajectory if item.get("assistant") is not None]),
            "native_harness_trajectory": str(workspace_path / "native_harness_trajectory.json"),
        }
        (workspace_path / "transcript.txt").write_text(transcript, encoding="utf-8")
        (workspace_path / "model_response_log.txt").write_text(
            "\n".join(json.dumps(item, ensure_ascii=False, default=str) for item in trajectory),
            encoding="utf-8",
        )
        (workspace_path / "native_harness_trajectory.json").write_text(
            json.dumps({"messages": messages, "turns": trajectory, "result": payload}, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        return payload

    def _client(self):
        if self.client is not None:
            return self.client
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"Set {self.api_key_env} before running Bayesian-Agent native harness tasks.")
        return OpenAIChatClient(
            api_key=api_key,
            model=self.model,
            base_url=self.base_url,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            timeout_seconds=self.timeout_seconds,
            verify_ssl=self.verify_ssl,
            host_header=self.host_header,
        )


def _assistant_message(content: str, tool_calls: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    message: Dict[str, Any] = {"role": "assistant", "content": content or ""}
    if tool_calls:
        message["tool_calls"] = []
        for index, call in enumerate(tool_calls):
            call_id = str(call.get("id") or f"call_{index}")
            name = str(call.get("name") or "")
            arguments = dict(call.get("arguments") or {})
            message["tool_calls"].append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(arguments, ensure_ascii=False, default=str),
                    },
                }
            )
    return message


def _accumulate_usage(totals: Dict[str, int], usage: Mapping[str, Any]) -> None:
    totals["input_tokens"] += int(usage.get("input_tokens") or 0)
    totals["output_tokens"] += int(usage.get("output_tokens") or 0)
    totals["total_tokens"] += int(usage.get("total_tokens") or 0)
