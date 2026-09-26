#!/usr/bin/env python3
# Copyright (c) TeamZedra.

from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from typing import Any, Optional

from aiohttp import web

# Configration

HOST = "127.0.0.1"
PORT = 1337

RESET_COMMANDS = {
    "/clear",
    "/reset",
    "/new",
    "/clearchat",
    "/resetchat",
    "/deletecurrentchat",
    "!clear",
    "!reset",
    "!new",
    "!clearchat",
    "!resetchat",
    "!deletecurrentchat",
    "?clear",
    "?reset",
    "?new",
    "?clearchat",
    "?resetchat",
    "?deletecurrentchat",
}

JOB_TIMEOUT = 600
# Budget for the whole prompt. Copilot's tool schemas + subagent list alone can
# run past the old 24000 easily (that's what silently dropped the edit_file
# tool out of the prompt before) - this only bounds how much conversation
# history gets kept now (see build_prompt / _truncate_conversation_to_budget),
# never the tool list or system context, so raise it if you still see
# conversation getting squeezed to the 2000-char floor in practice.
MAX_PROMPT_CHARS = 60000
HISTORY_TAIL = 12
ARG_CHUNK_SIZE = 24
AUTO_RESET_THRESHOLD = (
    150000  # auto-reset chat session if server tokens exceed this (0 to disable)
)

MODELS = [
    {
        "id": "deepseek-chat",
        "object": "model",
        "created": 0,
        "owned_by": "deepseek-web",
    },
    {
        "id": "deepseek-reasoner",
        "object": "model",
        "created": 0,
        "owned_by": "deepseek-web",
    },
]

active_ws: Optional[web.WebSocketResponse] = None
pending_jobs: dict[str, asyncio.Future] = {}
# Optional per-job queue for real-time streaming. A message from the browser
# with a "delta"/"reasoningDelta" key is routed here instead of resolving the
# job's future; the existing full-result message (no delta key) still
# resolves the future as before AND, if a queue is registered, pushes a
# sentinel (None) so a streaming consumer knows no more deltas are coming.
# This means an unmodified browser userscript (one that only ever sends the
# final message) degrades gracefully to "no deltas, just the final result" -
# nothing breaks if the userscript side isn't updated.
pending_delta_queues: dict[str, asyncio.Queue] = {}
_job_lock: Optional[asyncio.Lock] = None

# Tracks consecutive turns where every parsed tool call was dropped for
# missing required arguments (see missing_required_args). Used to detect
# and break the "model keeps emitting {} and the agent keeps re-prompting
# forever" loop, since neither side ever gets a signal to stop on its own.
_consecutive_invalid_tool_turns: int = 0
INVALID_TOOL_LOOP_THRESHOLD = 2  # force a hard-stop message after this many in a row


def job_lock() -> asyncio.Lock:
    #Lazily create the lock so import works without a running loop.
    global _job_lock
    if _job_lock is None:
        _job_lock = asyncio.Lock()
    return _job_lock


TOOL_PROTOCOL_TEMPLATE = """You are an AI assistant exposed through an OpenAI-compatible API with tool support.

You have access to the following tools:
{tool_list}

TOOL INSTRUCTIONS:
1. When you need to take an action (e.g. run a command, search files, read/write a file), output a tool call using this EXACT XML format:
<tool_call>
<name>TOOL_NAME</name>
<arguments>{{"argument_name": "argument value"}}</arguments>
</tool_call>

2. <arguments> must be a valid JSON object.
3. Use ONLY tool names from the available tools list above.
4. When you DO NOT need to call a tool (e.g. answering questions, greetings, explanations, math calculations, or providing the final answer after reviewing tool results), respond DIRECTLY in normal conversational text / markdown. When providing code, scripts, or terminal commands, ALWAYS format them inside markdown code fences with the language identifier (e.g. ```python ... ```). Do NOT use any <tool_call> tags.
5. NEVER use bash/echo/Write-Output just to speak to the user. Speak directly in your response text.
"""

CLINE_XML_PROTOCOL = """# Tool Use Formatting

You are an AI assistant integrated into Cline (a VS Code coding agent).

You MUST respond using ONLY the following XML tool format. Do NOT write plain prose answers.

## attempt_completion
When you are ready to give your final answer or complete the task, use:

<attempt_completion>
<result>
Your final answer or task summary goes here.
</result>
</attempt_completion>

## ask_followup_question
When you need more information from the user, use:

<ask_followup_question>
<question>Your question goes here.</question>
</ask_followup_question>

## Rules
1. Every response MUST be exactly one tool call block.
2. Do not wrap the XML in markdown code fences.
3. Do not include any text outside the XML block.
4. For simple factual questions, respond with attempt_completion containing the answer.
"""

CLINE_XML_MARKERS = (
    "<attempt_completion>",
    "<ask_followup_question>",
    "# Tool Use",
    "TOOL USE",
)


async def websocket_handler(request: web.Request) -> web.WebSocketResponse:
    global active_ws
    ws = web.WebSocketResponse(heartbeat=30, max_msg_size=16 * 1024 * 1024)
    await ws.prepare(request)

    active_ws = ws
    print("\033[92m[Bridge]\033[0m DeepSeek browser tab connected!")

    try:
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                except json.JSONDecodeError:
                    print(
                        f"\033[91m[Bridge]\033[0m Bad JSON from browser: {msg.data[:200]!r}"
                    )
                    continue
                req_id = data.get("id")
                queue = pending_delta_queues.get(req_id) if req_id else None
                if "delta" in data or "reasoningDelta" in data:
                    if queue is not None:
                        queue.put_nowait(data)
                    continue
                future = pending_jobs.get(req_id) if req_id else None
                if future and not future.done():
                    future.set_result(data)
                if queue is not None:
                    queue.put_nowait(None)  # sentinel: final result is ready
            elif msg.type in (web.WSMsgType.CLOSED, web.WSMsgType.ERROR):
                break
    finally:
        if active_ws == ws:
            active_ws = None
        for fut in list(pending_jobs.values()):
            if not fut.done():
                fut.set_result({"error": "Browser tab disconnected mid-request."})
        for queue in list(pending_delta_queues.values()):
            queue.put_nowait(None)
        print("\033[93m[Bridge]\033[0m DeepSeek browser tab disconnected.")

    return ws


def _message_text(message: dict) -> str:
    parts = []
    content = message.get("content", "")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif "text" in block:
                    parts.append(str(block["text"]))
    elif content:
        parts.append(str(content))
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        for tc in tool_calls:
            fn = tc.get("function", tc) if isinstance(tc, dict) else {}
            name = fn.get("name", "")
            args = fn.get("arguments", "")
            if isinstance(args, dict):
                args = json.dumps(args, ensure_ascii=False)
            if name:
                parts.append(
                    f"<tool_call>\n<name>{name}</name>\n<arguments>{args}</arguments>\n</tool_call>"
                )

    return "\n".join(p for p in parts if p)


def _truncate_middle(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    keep = limit // 2
    return text[:keep] + "\n\n[... truncated ...]\n\n" + text[-keep:]


def _truncate_conversation_to_budget(conversation: list[str], budget: int) -> list[str]:
    #Drop OLDEST conversation turns (not characters mid-string) until what's left
    #fits budget. This is the only part of the prompt allowed to be trimmed for
    #size - the tool protocol/system context must never be sliced, because a
    #raw middle-of-string cut can (and did) land inside a tool's JSON schema,
    #silently hiding whole tools (e.g. an edit_file tool) from the model.
    if budget <= 0:
        return []
    kept: list[str] = []
    total = 0
    for turn in reversed(conversation):
        turn_len = len(turn) + 2  # account for the "\n\n".join separator
        if kept and total + turn_len > budget:
            break
        kept.append(turn)
        total += turn_len
    kept.reverse()
    if len(kept) < len(conversation):
        kept.insert(0, "[... earlier turns truncated ...]")
    return kept


def _build_tool_list(tools: list[dict]) -> str:
    lines = []
    for tool in tools:
        fn = tool.get("function", tool) if isinstance(tool, dict) else {}
        name = fn.get("name", "unknown")
        desc = (fn.get("description") or "").strip().replace("\n", " ")
        params = fn.get("parameters") or {}
        lines.append(f"- {name}: {desc}")
        lines.append(f"  parameters: {json.dumps(params, ensure_ascii=False)}")
    return "\n".join(lines)


def _detect_cline_xml(messages: list[dict]) -> bool:
    for message in messages:
        if message.get("role") == "system":
            text = _message_text(message)
            if any(marker in text for marker in CLINE_XML_MARKERS):
                return True
    return False


def build_prompt(
    messages: list[dict],
    tools: Optional[list[dict]],
    tool_choice: Any = None,
) -> tuple[str, str]:
    use_tools = bool(tools) and tool_choice != "none"
    cline_xml = (not use_tools) and _detect_cline_xml(messages)

    if use_tools:
        mode = "native_tools"
    elif cline_xml:
        mode = "cline_xml"
    else:
        mode = "text"

    system_parts: list[str] = []
    conversation: list[str] = []

    for message in messages:
        role = (message.get("role") or "user").lower()
        text = _message_text(message)

        if role == "system":
            system_parts.append(text)
        elif role == "user":
            conversation.append(f"User:\n{text}")
        elif role == "assistant":
            if text:
                conversation.append(f"Assistant:\n{text}")
        elif role == "tool":
            conversation.append(f"Tool result:\n{text}")
    if len(conversation) > HISTORY_TAIL:
        conversation = conversation[-HISTORY_TAIL:]

    # Tool protocol and system context are load-bearing: a tool the model was
    # never told about is a tool it can never call, so these are built first
    # and are exempt from character-budget truncation. Only the conversation
    # history (already message-capped by HISTORY_TAIL above) gets trimmed
    # further, and only by dropping whole oldest turns - never by slicing
    # through the middle of any section's raw text.
    fixed_sections: list[str] = []

    if mode == "native_tools":
        fixed_sections.append(
            TOOL_PROTOCOL_TEMPLATE.format(tool_list=_build_tool_list(tools or []))
        )
    elif mode == "cline_xml":
        fixed_sections.append(CLINE_XML_PROTOCOL)

    if system_parts:
        fixed_sections.append("# System Context\n" + "\n\n".join(system_parts))

    fixed_prompt = "\n\n".join(fixed_sections).strip()

    if fixed_prompt and len(fixed_prompt) >= MAX_PROMPT_CHARS:
        # The tool list + system context alone already exceed the budget.
        # Ship them whole regardless (an oversized-but-complete tool list
        # beats a truncated one with tools silently missing) and give the
        # conversation section a small fixed floor so the model still has
        # some recent context to work from.
        conversation_budget = min(2000, MAX_PROMPT_CHARS // 4)
    else:
        conversation_budget = MAX_PROMPT_CHARS - len(fixed_prompt)

    conversation = _truncate_conversation_to_budget(conversation, conversation_budget)

    sections = list(fixed_sections)
    if conversation:
        sections.append("# Conversation\n" + "\n\n".join(conversation))

    prompt = "\n\n".join(sections).strip()
    return prompt, mode


_TOOL_CALL_BLOCK_RE = re.compile(
    r"<tool_call((?:\s+[^>]*)?)>(.*?)(?:</tool_call>|</[|\uFF5C]{2}DSML[|\uFF5C]{2}\s*(?:calls|invoke|parameter)>|(?=<tool_call(?:\s|>))|$)",
    re.DOTALL | re.IGNORECASE,
)
# Attribute-style dialect some models (and Claude-family harnesses) fall back
# to instead of the <n>/<arguments> form we ask for:
#   <tool_call name="TOOL_NAME">
#     <parameter name="PARAM">VALUE</parameter>
#     ...
#   </tool_call>
_TOOL_CALL_NAME_ATTR_RE = re.compile(
    r"\bname=[\"']([^\"']+)[\"']", re.IGNORECASE
)
_PARAMETER_TAG_RE = re.compile(
    r"<parameter\s+name=[\"']([^\"']+)[\"'][^>]*>(.*?)</parameter>",
    re.DOTALL | re.IGNORECASE,
)
_TAG_NAME_RE = re.compile(r"<name>(.*?)</name>", re.DOTALL | re.IGNORECASE)
_TAG_ARGS_RE = re.compile(
    r"<arguments>(.*?)(?:</arguments>|</[|\uFF5C]{2}DSML[|\uFF5C]{2}\s*(?:parameter|invoke|calls)>|$)",
    re.DOTALL | re.IGNORECASE,
)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
_FENCE_RE = re.compile(r"```[a-zA-Z0-9_-]*\s*|\s*```")
_DSML_INVOKE_RE = re.compile(
    r"<[|\uFF5C]{2}DSML[|\uFF5C]{2}\s+invoke\s+name=[\"']([^\"']+)[\"']\s*>(.*?)(?:</[|\uFF5C]{2}DSML[|\uFF5C]{2}\s+invoke>|$)",
    re.DOTALL | re.IGNORECASE,
)
_DSML_PARAM_RE = re.compile(
    r"<[|\uFF5C]{2}DSML[|\uFF5C]{2}\s+parameter\s+name=[\"']([^\"']+)[\"'](?:\s+[^>]*)?>(.*?)(?:</[|\uFF5C]{2}DSML[|\uFF5C]{2}\s+parameter>|$)",
    re.DOTALL | re.IGNORECASE,
)
# DeepSeek sometimes fuses a parameter's closing tag with the NEXT
# parameter's opening tag into one hybrid token, e.g.:
#   ...Core.h</｜｜DSML｜｜ parameter name="isRegexp" string="false">true</｜｜DSML｜｜ parameter name="query" ...
# i.e. no standalone "</...parameter>" ever appears, so _DSML_PARAM_RE's
# finditer finds zero matches and the whole block leaks through as text.
# This generic token scanner matches ANY DSML boundary marker - open, close,
# or the malformed hybrid of both - and reconstructs calls from the marker
# sequence rather than requiring cleanly paired open/close tags.
_DSML_TOKEN_RE = re.compile(
    r"<(/?)[|\uFF5C]{2}DSML[|\uFF5C]{2}\s*(calls|invoke|parameter)?(?:\s+name=[\"']([^\"']+)[\"'])?[^>]*>",
    re.IGNORECASE,
)


def _coerce_dsml_value(pval: str) -> Any:
    pval = pval.strip()
    if pval.lower() == "true":
        return True
    if pval.lower() == "false":
        return False
    if pval.lower() == "null":
        return None
    if pval.isdigit():
        return int(pval)
    try:
        return json.loads(pval)
    except Exception:
        return pval


def _parse_dsml_tool_calls(raw: str) -> list[dict]:
    #Reconstructs DSML tool calls from a stream of boundary markers instead of
    #requiring well-paired open/close tags - tolerates the fused-tag dialect
    #above as well as the well-formed one.
    calls: list[dict] = []
    invoke_name: Optional[str] = None
    args: dict[str, Any] = {}
    param_name: Optional[str] = None
    param_start: Optional[int] = None

    def close_param(end: int) -> None:
        nonlocal param_name, param_start
        if param_name is not None and param_start is not None:
            args[param_name] = _coerce_dsml_value(raw[param_start:end])
        param_name = None
        param_start = None

    def close_invoke() -> None:
        nonlocal invoke_name, args
        if invoke_name is not None:
            calls.append({"name": invoke_name, "arguments": args})
        invoke_name = None
        args = {}

    for m in _DSML_TOKEN_RE.finditer(raw):
        is_close = bool(m.group(1))
        kind = (m.group(2) or "").lower()
        name = m.group(3)

        if kind == "parameter":
            close_param(m.start())
            if name:
                param_name = name
                param_start = m.end()
            continue
        if kind == "invoke":
            close_param(m.start())
            if is_close:
                close_invoke()
            elif name:
                close_invoke()  # defensive: finalize any unterminated prior invoke
                invoke_name = name
                args = {}
            continue
        if kind == "calls":
            close_param(m.start())
            if is_close:
                close_invoke()
            continue
        # Unrecognized DSML token shape - ignore, don't disturb state.

    close_param(len(raw))
    close_invoke()
    return calls

# Copilot / other agents sometimes skip the <tool_call> wrapper entirely and
# emit the tool name itself as the root tag, e.g.:
#   <create_file>
#     <filePath>...</filePath>
#     <content>...</content>
#   </create_file>
_CHILD_TAG_RE = re.compile(r"<([a-zA-Z_][\w-]*)>(.*?)</\1>", re.DOTALL)


def _parse_direct_xml_tool_calls(raw: str, tool_names: list[str]) -> list[dict]:
    #Parse <tool_name><param>val</param>...</tool_name> blocks, one root tag per known tool.
    if not raw or not tool_names:
        return []

    calls: list[dict] = []
    for name in tool_names:
        if name in ("tool_call", "arguments", "name"):
            continue
        safe_name = re.escape(name)
        block_re = re.compile(
            rf"<{safe_name}>(.*?)</{safe_name}>", re.DOTALL | re.IGNORECASE
        )
        for block_match in block_re.finditer(raw):
            body = block_match.group(1)
            arguments: dict[str, Any] = {}
            for child_match in _CHILD_TAG_RE.finditer(body):
                key = child_match.group(1).strip()
                val = child_match.group(2)
                # Strip exactly one leading/trailing newline (formatting
                # artifact) but preserve internal whitespace/content, e.g.
                # a file's actual contents.
                if val.startswith("\n"):
                    val = val[1:]
                if val.endswith("\n"):
                    val = val[:-1]
                arguments[key] = val
            if arguments:
                calls.append({"name": name, "arguments": arguments})

    return calls


def _strip_fences(text: str) -> str:
    return _FENCE_RE.sub("", text).strip()


def _ensure_code_fenced(text: str) -> str:
    #If the model's text response is purely code but lacks markdown fences, wrap it
    trimmed = text.strip()
    if "```" in trimmed or not trimmed:
        return text

    code_indicators = (
        (
            "python",
            (
                "with open(",
                "import ",
                "from ",
                "def ",
                "class ",
                "print(",
                "if __name__",
            ),
        ),
        (
            "powershell",
            (
                "Get-",
                "Set-",
                "New-Item",
                "Remove-Item",
                "Write-Output",
                "Test-Path",
            ),
        ),
        (
            "bash",
            (
                "#!/bin/bash",
                "#!/bin/sh",
                "sudo ",
                "curl ",
                "npm ",
                "git ",
                "pip ",
                "docker ",
            ),
        ),
        (
            "javascript",
            (
                "const ",
                "let ",
                "var ",
                "function(",
                "function ",
                "console.log(",
                "export ",
            ),
        ),
    )

    first_line = trimmed.split("\n", 1)[0].strip()
    for lang, markers in code_indicators:
        if any(first_line.startswith(m) for m in markers):
            return f"```{lang}\n{trimmed}\n```"

    lines = trimmed.split("\n")
    if len(lines) >= 2:
        if any(
            l.strip().startswith(
                ("import ", "from ", "def ", "class ", "with open(", "with ")
            )
            for l in lines[:3]
        ):
            return f"```python\n{trimmed}\n```"

    return text


def _loads_lenient(raw: str) -> Optional[Any]:
    if raw is None:
        return None
    candidate = raw.strip()
    if not candidate:
        return None

    def _escape_newlines_in_strings(text: str) -> str:
        result: list[str] = []
        in_string = False
        i = 0
        while i < len(text):
            c = text[i]
            if c == "\\" and in_string:
                # Keep escape + next char verbatim (handles \\, \", etc.).
                result.append(c)
                i += 1
                if i < len(text):
                    result.append(text[i])
                    i += 1
                continue
            if c == '"':
                in_string = not in_string
                result.append(c)
            elif in_string and c == "\n":
                result.append("\\n")
            elif in_string and c == "\r":
                if i + 1 < len(text) and text[i + 1] == "\n":
                    i += 1  # consume the paired \n; emit a single \n
                result.append("\\n")
            else:
                result.append(c)
            i += 1
        return "".join(result)

    no_trail = re.sub(r",\s*([}\]])", r"\1", candidate)
    escaped = _escape_newlines_in_strings(candidate)
    escaped_no_trail = re.sub(r",\s*([}\]])", r"\1", escaped)

    for attempt in (candidate, no_trail, escaped, escaped_no_trail):
        try:
            return json.loads(attempt)
        except (json.JSONDecodeError, ValueError):
            continue
    return None


def _find_json_in(text: str) -> Optional[dict]:
    for match in _JSON_OBJECT_RE.finditer(text):
        parsed = _loads_lenient(match.group(0))
        if isinstance(parsed, dict):
            return parsed
    return None


def parse_tool_calls(raw: str, tool_names: list[str]) -> list[dict]:
    if not raw:
        return []

    calls: list[dict] = []

    for open_attrs, block in _TOOL_CALL_BLOCK_RE.findall(raw):
        block = block.strip()
        if not block and not open_attrs:
            continue

        # Attribute-style dialect: <tool_call name="X"><parameter name="Y">Z</parameter></tool_call>
        attr_name_match = _TOOL_CALL_NAME_ATTR_RE.search(open_attrs) if open_attrs else None
        param_matches = _PARAMETER_TAG_RE.findall(block) if block else []
        if attr_name_match and param_matches:
            name = attr_name_match.group(1).strip()
            arguments = {}
            for pname, pval in param_matches:
                pval = pval.strip()
                if pval.startswith("\n"):
                    pval = pval[1:]
                if pval.endswith("\n"):
                    pval = pval[:-1]
                arguments[pname.strip()] = pval
            calls.append({"name": name, "arguments": arguments})
            continue

        if not block:
            continue
        name_match = _TAG_NAME_RE.search(block)
        args_match = _TAG_ARGS_RE.search(block)

        if name_match:
            name = name_match.group(1).strip()
            args_raw = args_match.group(1).strip() if args_match else ""
            arguments = _loads_lenient(args_raw) if args_raw else None
            if not isinstance(arguments, dict):
                arguments = _find_json_in(block) or (
                    _find_json_in(args_raw) if args_raw else None
                )
            if not isinstance(arguments, dict):
                arguments = {"result": _strip_fences(args_raw)} if args_raw else {}
            calls.append({"name": name, "arguments": arguments})
            continue
        payload = _find_json_in(block)
        if isinstance(payload, dict):
            name = payload.get("name") or payload.get("tool") or payload.get("function")
            args = (
                payload.get("arguments")
                or payload.get("parameters")
                or payload.get("args")
            )
            if isinstance(args, str):
                args = _loads_lenient(args)
            if name:
                calls.append(
                    {
                        "name": str(name),
                        "arguments": (
                            args if isinstance(args, dict) else {"result": args}
                        ),
                    }
                )

    if calls:
        return _validate_calls(calls, tool_names)

    # Copilot-style direct XML: <tool_name><param>val</param></tool_name>
    # with no <tool_call> wrapper at all.
    direct_calls = _parse_direct_xml_tool_calls(raw, tool_names)
    if direct_calls:
        return _validate_calls(direct_calls, tool_names)

    # DeepSeek Native DSML format - well-formed:
    # <｜｜DSML｜｜ calls>
    # <｜｜DSML｜｜ invoke name="TOOL_NAME">
    # <｜｜DSML｜｜ parameter name="PARAM" string="true">VALUE</｜｜DSML｜｜ parameter>
    # </｜｜DSML｜｜ invoke>
    # </｜｜DSML｜｜ calls>
    # ...and the fused/malformed variant where a parameter's close and the
    # next parameter's open get merged into one hybrid tag. The tolerant
    # token scanner (_parse_dsml_tool_calls) handles both shapes uniformly -
    # it's used as the sole DSML parser rather than trying the strict
    # regex pair first, because that strict pair's own end-of-string
    # fallback can "successfully" match the malformed case by swallowing
    # everything after the first parameter into one value, which would
    # otherwise mask the real per-parameter split done here.
    if "DSML" in raw:
        dsml_calls = _parse_dsml_tool_calls(raw)
        if dsml_calls:
            return _validate_calls(dsml_calls, tool_names)
    payload = _find_json_in(_strip_fences(raw))
    if isinstance(payload, dict):
        name = payload.get("name") or payload.get("tool") or payload.get("function")
        args = (
            payload.get("arguments") or payload.get("parameters") or payload.get("args")
        )
        if isinstance(args, str):
            args = _loads_lenient(args)
        if name:
            return _validate_calls(
                [
                    {
                        "name": str(name),
                        "arguments": (
                            args if isinstance(args, dict) else {"result": args}
                        ),
                    }
                ],
                tool_names,
            )

    return []


def _validate_calls(calls: list[dict], tool_names: list[str]) -> list[dict]:
    valid = []
    for call in calls:
        name = call["name"]
        if tool_names and name not in tool_names:
            tail = name.split(".")[-1]
            if tail in tool_names:
                name = tail
            else:
                continue
        valid.append({"name": name, "arguments": call["arguments"]})
    return valid


def coerce_arguments(arguments: dict, tool_schema: Optional[dict]) -> dict:
    if not isinstance(arguments, dict):
        return {"result": str(arguments)}

    if not tool_schema:
        return arguments

    properties = tool_schema.get("properties") or {}
    required = tool_schema.get("required") or []
    if all(key in arguments for key in required):
        return arguments
    if len(required) == 1:
        key = required[0]
        if key not in arguments and arguments:
            if len(arguments) == 1:
                arguments = {key: next(iter(arguments.values()))}
            else:
                text_keys = ("result", "text", "content", "response", "answer", "value")
                for candidate in text_keys:
                    if candidate in arguments:
                        arguments = {key: arguments[candidate]}
                        break
    if properties and not tool_schema.get("additionalProperties", True):
        arguments = {k: v for k, v in arguments.items() if k in properties}

    return arguments


def missing_required_args(arguments: dict, tool_schema: Optional[dict]) -> list[str]:
    #Return the required schema keys that arguments still lacks after coercion.
    #This is what catches the "model emits <tool_call> with {} arguments"
    #failure mode: rather than shipping a call we already know is invalid and
    #letting the client bounce it back for the model to blindly retry, we
    #catch it here and fall back to plain text in the same turn.
    if not tool_schema:
        return []
    required = tool_schema.get("required") or []
    if not isinstance(arguments, dict):
        return list(required)
    return [key for key in required if key not in arguments]


def _strip_tool_call_artifacts(raw: str, tool_names: list[str]) -> str:
    #Remove leftover tool-call XML (any of the three formats we parse) so a
    #dropped/invalid call doesn't leak raw markup into a text fallback.
    text = _TOOL_CALL_BLOCK_RE.sub("", raw)
    text = _DSML_INVOKE_RE.sub("", text)
    # Models sometimes double up the closing tag (e.g. "...</tool_call> </tool_call>");
    # the block regex only consumes the first one, so mop up any stragglers.
    text = re.sub(r"</tool_call>", "", text, flags=re.IGNORECASE)
    # Any leftover DSML boundary markers - including the fused/malformed
    # tags _DSML_INVOKE_RE/_DSML_PARAM_RE won't match on their own - so a
    # dropped call doesn't leak raw "｜｜DSML｜｜" markup into the text.
    text = _DSML_TOKEN_RE.sub("", text)
    for name in tool_names:
        if name in ("tool_call", "arguments", "name"):
            continue
        safe_name = re.escape(name)
        text = re.sub(
            rf"<{safe_name}>.*?</{safe_name}>", "", text, flags=re.DOTALL | re.IGNORECASE
        )
    return text.strip()


def chunk(
    completion_id: str,
    created: int,
    model: str,
    delta: dict,
    finish_reason: Optional[str] = None,
) -> dict:
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }


def usage_payload(
    prompt: str, completion: str, accumulated_tokens: int = 0, initial_tokens: int = 0
) -> dict:
    if accumulated_tokens > 0:
        total = accumulated_tokens
        completion_tokens = (
            max(1, accumulated_tokens - initial_tokens)
            if initial_tokens > 0
            else max(1, len(completion) // 4)
        )
        prompt_tokens = max(1, total - completion_tokens)
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total,
        }
    return {
        "prompt_tokens": max(1, len(prompt) // 4),
        "completion_tokens": max(1, len(completion) // 4),
        "total_tokens": max(1, (len(prompt) + len(completion)) // 4),
    }


async def _trigger_browser_reset():
    await asyncio.sleep(0.5)
    async with job_lock():
        if active_ws and not active_ws.closed:
            try:
                await active_ws.send_str(
                    json.dumps(
                        {
                            "action": "delete_chat",
                            "id": f"auto-reset-{int(time.time())}",
                        }
                    )
                )
            except Exception:
                pass


async def sse_write(response: web.StreamResponse, payload: dict) -> None:
    data = json.dumps(payload, ensure_ascii=False)
    await response.write(f"data: {data}\n\n".encode("utf-8"))


def iter_text_fragments(text: str, size: int = ARG_CHUNK_SIZE):
    for index in range(0, len(text), size):
        yield text[index : index + size]


_last_job_time: float = 0.0
MIN_JOB_INTERVAL: float = 1.5  # seconds between submissions to prevent rate limiting


async def run_browser_job(prompt: str, thinking: bool) -> dict:
    global _last_job_time
    async with job_lock():
        if not active_ws or active_ws.closed:
            return {"error": "DeepSeek browser tab is not connected."}
        now = time.time()
        elapsed = now - _last_job_time
        if elapsed < MIN_JOB_INTERVAL:
            await asyncio.sleep(MIN_JOB_INTERVAL - elapsed)

        req_id = f"{int(time.time())}-{uuid.uuid4().hex[:6]}"
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        pending_jobs[req_id] = future

        try:
            await active_ws.send_str(
                json.dumps(
                    {
                        "id": req_id,
                        "prompt": prompt,
                        "thinkingEnabled": bool(thinking),
                    }
                )
            )
            res = await asyncio.wait_for(future, timeout=JOB_TIMEOUT)
            _last_job_time = time.time()
            return res
        except asyncio.TimeoutError:
            _last_job_time = time.time()
            return {"error": "DeepSeek generation timed out."}
        except Exception as e:
            # A page refresh/navigation on the DeepSeek tab can leave `active_ws`
            # pointing at a half-dead connection for a moment before
            # websocket_handler's own close detection catches up and clears it -
            # send_str (or the transport under it) can raise here instead of the
            # tab cleanly resolving pending_jobs with an error. Without this,
            # the exception escapes run_browser_job entirely, bypasses the
            # normal "error" in result handling below, and surfaces to VS Code
            # as a raw unhandled 500 instead of a readable error message.
            _last_job_time = time.time()
            print(f"\033[91m[Bridge]\033[0m run_browser_job send/transport error: {e!r}")
            return {"error": f"Lost connection to the DeepSeek browser tab: {e}"}
        finally:
            pending_jobs.pop(req_id, None)


async def stream_browser_job(prompt: str, thinking: bool):
    #Async generator: yields ("delta", data) as partial text/reasoning
    #arrives, then a final ("final", result_dict). If the browser userscript
    #doesn't send delta messages, this degrades to a single ("final", ...)
    #yield with no deltas in between - same effective behavior as
    #run_browser_job, just through the same code path callers can rely on.
    global _last_job_time
    async with job_lock():
        if not active_ws or active_ws.closed:
            yield ("final", {"error": "DeepSeek browser tab is not connected."})
            return

        now = time.time()
        elapsed = now - _last_job_time
        if elapsed < MIN_JOB_INTERVAL:
            await asyncio.sleep(MIN_JOB_INTERVAL - elapsed)

        req_id = f"{int(time.time())}-{uuid.uuid4().hex[:6]}"
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        queue: asyncio.Queue = asyncio.Queue()
        pending_jobs[req_id] = future
        pending_delta_queues[req_id] = queue

        try:
            await active_ws.send_str(
                json.dumps(
                    {
                        "id": req_id,
                        "prompt": prompt,
                        "thinkingEnabled": bool(thinking),
                    }
                )
            )
            deadline = time.time() + JOB_TIMEOUT
            while True:
                remaining = deadline - time.time()
                if remaining <= 0:
                    yield ("final", {"error": "DeepSeek generation timed out."})
                    return
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=remaining)
                except asyncio.TimeoutError:
                    yield ("final", {"error": "DeepSeek generation timed out."})
                    return
                if item is None:
                    break  # sentinel: the final result is ready on the future
                yield ("delta", item)

            _last_job_time = time.time()
            final_result = (
                future.result()
                if future.done()
                else {"error": "No final result received."}
            )
            yield ("final", final_result)
        except Exception as e:
            # Same rationale as run_browser_job's except Exception branch: a
            # DeepSeek tab refresh mid-stream can raise out of send_str/the
            # transport instead of resolving gracefully, and this generator is
            # driven directly by _stream_text_reply with no outer try/except -
            # letting this escape would abort the whole SSE response mid-write
            # rather than yielding a clean final error the caller already knows
            # how to render.
            _last_job_time = time.time()
            print(f"\033[91m[Bridge]\033[0m stream_browser_job send/transport error: {e!r}")
            yield ("final", {"error": f"Lost connection to the DeepSeek browser tab: {e}"})
        finally:
            pending_jobs.pop(req_id, None)
            pending_delta_queues.pop(req_id, None)


CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, Authorization",
}


def error_response(message: str, status: int, err_type: str = "invalid_request_error"):
    return web.json_response(
        {"error": {"message": message, "type": err_type, "code": None}},
        status=status,
        headers=CORS_HEADERS,
    )


async def handle_models(request: web.Request) -> web.Response:
    if request.method == "OPTIONS":
        return web.Response(status=204, headers=CORS_HEADERS)
    return web.json_response({"object": "list", "data": MODELS}, headers=CORS_HEADERS)


async def handle_health(request: web.Request) -> web.Response:
    return web.json_response(
        {
            "status": "ok",
            "browser_connected": bool(active_ws and not active_ws.closed),
            "pending_jobs": len(pending_jobs),
        },
        headers=CORS_HEADERS,
    )


def _is_cline_request(
    request: web.Request,
    messages: Optional[list[dict]] = None,
) -> bool:
    # Cline SSE parser fails on tool_calls chunks, so force non-streaming.
    ua = request.headers.get("User-Agent", "")
    if "cline" in ua.lower():
        return True
    if request.headers.get("X-Cline-Version"):
        return True
    if messages and _detect_cline_xml(messages):
        return True
    return False


async def _simple_text_response(
    request: web.Request, model: str, stream: bool, reply_text: str
) -> web.Response:
    #Build a short, definitive assistant reply with no tool_calls, bypassing
    #the browser entirely. Used for slash-commands and for short-circuiting
    #agent auto-continuation stubs (see CONTINUATION_STUB_RE).
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    if not stream:
        return web.json_response(
            {
                "id": completion_id,
                "object": "chat.completion",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": reply_text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": usage_payload("", reply_text),
            },
            headers=CORS_HEADERS,
        )

    response = web.StreamResponse(
        status=200,
        headers={
            **CORS_HEADERS,
            "Content-Type": "text/event-stream; charset=utf-8",
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
        },
    )
    await response.prepare(request)
    await sse_write(
        response,
        chunk(completion_id, created, model, {"role": "assistant", "content": reply_text}),
    )
    await sse_write(response, chunk(completion_id, created, model, {}, "stop"))
    await response.write(b"data: [DONE]\n\n")
    await response.write_eof()
    return response


# Agent harnesses (VS Code Copilot Agent Mode among them) sometimes keep a
# task "alive" by sending a synthetic follow-up turn that looks like a user
# message but isn't one - e.g. "Continue.", "Continue to iterate?", "Proceed."
# with no real new instruction. If the model already signaled it was done
# last turn, forwarding one of these stubs to DeepSeek just invites another
# round of self-directed "work" and the loop never actually ends. We detect
# both sides of that pattern and refuse to continue, forcing a genuine stop.
CONTINUATION_STUB_RE = re.compile(
    r"^(continue\.?|continue to iterate\??|proceed\.?|go ahead\.?|keep going\.?|next\.?)$",
    re.IGNORECASE,
)
COMPLETION_SIGNAL_RE = re.compile(
    r"\b(task is complete|all done|nothing (else|more) to do|"
    r"i'?ve finished|completed successfully|that completes the|"
    r"no further (action|steps?) (is |are )?needed)\b",
    re.IGNORECASE,
)
_last_turn_was_final: bool = False


def _looks_like_continuation_stub(text: str) -> bool:
    stripped = text.strip()
    return (not stripped) or bool(CONTINUATION_STUB_RE.match(stripped))


async def _stream_text_reply(
    request: web.Request,
    model: str,
    prompt: str,
    is_reasoner: bool,
    completion_id: str,
    created: int,
    include_usage: bool,
) -> web.Response:
    #Forward reasoning/content deltas live via SSE as DeepSeek generates them.
    #Degrades gracefully to a single chunked dump of the full text if the
    #browser userscript never sends delta messages (same end result as the
    #old post-hoc chunking, just reached through this path instead).
    global _last_turn_was_final

    response = web.StreamResponse(
        status=200,
        reason="OK",
        headers={
            **CORS_HEADERS,
            "Content-Type": "text/event-stream; charset=utf-8",
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
    await response.prepare(request)

    first = True
    got_any_delta = False
    streamed_parts: list[str] = []
    final_result: dict = {}

    async for kind, item in stream_browser_job(prompt, is_reasoner):
        if kind == "delta":
            got_any_delta = True
            reasoning_delta = item.get("reasoningDelta")
            text_delta = item.get("delta")
            if reasoning_delta:
                delta = {"reasoning_content": reasoning_delta}
                if first:
                    delta["role"] = "assistant"
                    first = False
                await sse_write(response, chunk(completion_id, created, model, delta))
            if text_delta:
                streamed_parts.append(text_delta)
                delta = {"content": text_delta}
                if first:
                    delta["role"] = "assistant"
                    first = False
                await sse_write(response, chunk(completion_id, created, model, delta))
        else:
            final_result = item

    if "error" in final_result:
        # SSE headers are already sent, so we can't fall back to a JSON error
        # response - surface it as a final content chunk instead.
        err_text = f"\n\n[proxy error: {final_result['error']}]"
        delta = {"content": err_text}
        if first:
            delta["role"] = "assistant"
            first = False
        await sse_write(response, chunk(completion_id, created, model, delta))
        await sse_write(response, chunk(completion_id, created, model, {}, "stop"))
        await response.write(b"data: [DONE]\n\n")
        await response.write_eof()
        return response

    full_text = (final_result.get("text") or "".join(streamed_parts)).strip()
    reasoning = (final_result.get("reasoning") or "").strip()

    if not got_any_delta and full_text:
        # Legacy userscript: nothing streamed live, so fall back to chunking
        # the complete result exactly as the old code path did.
        for fragment in iter_text_fragments(reasoning, size=64):
            d = {"reasoning_content": fragment}
            if first:
                d["role"] = "assistant"
                first = False
            await sse_write(response, chunk(completion_id, created, model, d))
        for fragment in iter_text_fragments(_ensure_code_fenced(full_text), size=64):
            d = {"content": fragment}
            if first:
                d["role"] = "assistant"
                first = False
            await sse_write(response, chunk(completion_id, created, model, d))

    if first:
        await sse_write(
            response,
            chunk(completion_id, created, model, {"role": "assistant", "content": ""}),
        )

    _last_turn_was_final = bool(full_text and COMPLETION_SIGNAL_RE.search(full_text))

    await sse_write(response, chunk(completion_id, created, model, {}, "stop"))

    accumulated_tokens = int(final_result.get("accumulatedTokens") or 0)
    initial_tokens = int(final_result.get("initialTokens") or 0)

    if include_usage:
        usage = usage_payload(prompt, full_text, accumulated_tokens, initial_tokens)
        await sse_write(
            response,
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [],
                "usage": usage,
            },
        )

    if AUTO_RESET_THRESHOLD > 0 and accumulated_tokens >= AUTO_RESET_THRESHOLD:
        asyncio.create_task(_trigger_browser_reset())

    await response.write(b"data: [DONE]\n\n")
    await response.write_eof()
    return response


async def handle_chat_completions(request: web.Request) -> web.Response:
    if request.method == "OPTIONS":
        return web.Response(status=204, headers=CORS_HEADERS)

    try:
        body = await request.json()
    except (json.JSONDecodeError, ValueError):
        return error_response("Request body must be valid JSON.", 400)

    messages = body.get("messages") or []
    if not isinstance(messages, list) or not messages:
        return error_response("'messages' must be a non-empty array.", 400)

    model = body.get("model") or "deepseek-chat"
    stream = bool(body.get("stream", False))

    last_user_text = ""
    for m in reversed(messages):
        if (m.get("role") or "").lower() == "user":
            last_user_text = _message_text(m).strip()
            break

    global _last_turn_was_final
    if _last_turn_was_final and _looks_like_continuation_stub(last_user_text):
        print(
            f"\033[93m[Loop-Guard]\033[0m Refusing auto-continuation stub "
            f"{last_user_text!r} after a completed turn."
        )
        return await _simple_text_response(
            request,
            model,
            stream,
            "This task is already complete - standing by for your next instruction.",
        )

    if last_user_text.lower() in RESET_COMMANDS:
        _last_turn_was_final = False
        print(f"\033[93m[Command]\033[0m Reset chat triggered: {last_user_text}")
        async with job_lock():
            if active_ws and not active_ws.closed:
                try:
                    await active_ws.send_str(
                        json.dumps(
                            {
                                "action": "delete_chat",
                                "id": f"reset-{int(time.time())}",
                            }
                        )
                    )
                except Exception:
                    pass

        reply_text = "Chat session cleared. Started a fresh conversation."
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())

        if not stream:
            return web.json_response(
                {
                    "id": completion_id,
                    "object": "chat.completion",
                    "created": created,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": reply_text},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": usage_payload(last_user_text, reply_text),
                },
                headers=CORS_HEADERS,
            )

        response = web.StreamResponse(
            status=200,
            headers={
                **CORS_HEADERS,
                "Content-Type": "text/event-stream; charset=utf-8",
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
            },
        )
        await response.prepare(request)
        await sse_write(
            response,
            chunk(
                completion_id,
                created,
                model,
                {"role": "assistant", "content": reply_text},
            ),
        )
        await sse_write(
            response,
            chunk(completion_id, created, model, {}, "stop"),
        )
        await response.write(b"data: [DONE]\n\n")
        await response.write_eof()
        return response

    tools = body.get("tools")
    tool_choice = body.get("tool_choice")
    stream_options = body.get("stream_options") or {}
    include_usage = bool(stream_options.get("include_usage"))
    is_reasoner = "reasoner" in model.lower() or "r1" in model.lower()

    # Cline's SSE parser cannot handle tool_calls delta chunks, so we force non-streaming.
    # Detection is done after messages is parsed so we can check the system prompt.
    is_cline = _is_cline_request(request, messages)
    if is_cline and stream:
        stream = False
        print(
            "\033[93m[Request]\033[0m Cline detected! downgrading to non-streaming mode"
        )

    tool_names: list[str] = []
    tool_schemas: dict[str, dict] = {}
    if isinstance(tools, list):
        for tool in tools:
            fn = tool.get("function", tool) if isinstance(tool, dict) else {}
            name = fn.get("name")
            if name:
                tool_names.append(name)
                tool_schemas[name] = fn.get("parameters") or {}

    prompt, mode = build_prompt(
        messages, tools if isinstance(tools, list) else None, tool_choice
    )

    print(
        f"\033[96m[Request]\033[0m model={model} stream={stream} "
        f"mode={mode} tools={len(tool_names)} reasoner={is_reasoner} cline={is_cline}"
    )

    if stream and mode == "text":
        # No tools in play, so there's no "is this a tool call?" ambiguity to
        # resolve first - safe to forward text as DeepSeek generates it
        # instead of waiting for the full reply and chunking it after the
        # fact. Falls back gracefully to the old "chunk the whole thing"
        # behavior if the browser userscript doesn't send delta messages.
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())
        return await _stream_text_reply(
            request, model, prompt, is_reasoner, completion_id, created, include_usage
        )

    result = await run_browser_job(prompt, is_reasoner)

    if "error" in result:
        return error_response(result["error"], 502, "server_error")

    raw_text = (result.get("text") or "").strip()
    reasoning = (result.get("reasoning") or "").strip()

    print(f"\033[93m[Debug]\033[0m raw_text repr (first 300): {repr(raw_text[:300])}")

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    tool_calls: list[dict] = []
    content: str = ""

    if mode == "native_tools":
        parsed = parse_tool_calls(raw_text, tool_names)
        print(
            f"\033[93m[Debug]\033[0m parse_tool_calls: found={len(parsed)} "
            f"names={[c['name'] for c in parsed]} valid_names={tool_names}"
        )
        if ("<tool_call" in raw_text or "DSML" in raw_text) and not parsed:
            print(
                f"\033[91m[Debug-ParseError]\033[0m tool tag found but failed to parse: {repr(raw_text[:600])}"
            )
        elif not parsed:
            print(
                f"\033[94m[Debug]\033[0m No tool call! delivering natural text response"
            )

        if parsed:
            dropped: list[str] = []
            for call in parsed:
                arguments = coerce_arguments(
                    call["arguments"], tool_schemas.get(call["name"])
                )
                missing = missing_required_args(
                    arguments, tool_schemas.get(call["name"])
                )
                if missing:
                    dropped.append(f"{call['name']} (missing: {', '.join(missing)})")
                    continue
                tool_calls.append(
                    {
                        "id": f"call_{uuid.uuid4().hex[:24]}",
                        "index": len(tool_calls),
                        "name": call["name"],
                        "arguments": arguments,
                    }
                )
            if dropped:
                print(
                    f"\033[91m[Debug-InvalidArgs]\033[0m dropped incomplete tool call(s): "
                    f"{'; '.join(dropped)}"
                )
            if not tool_calls:
                # Every parsed call was missing required arguments (e.g. the
                # model emitted "{}"). Shipping any of these guarantees a
                # client-side error and invites the model to blindly retry
                # the same broken call. Deliver the raw text instead so the
                # turn isn't wasted, and the model gets a fresh chance next
                # message rather than repeating the same empty-args call.
                content = _ensure_code_fenced(raw_text)
        elif "attempt_completion" in tool_names:
            tool_calls.append(
                {
                    "id": f"call_{uuid.uuid4().hex[:24]}",
                    "index": 0,
                    "name": "attempt_completion",
                    "arguments": {
                        "result": _ensure_code_fenced(raw_text) or "(empty response)"
                    },
                }
            )
        else:
            content = _ensure_code_fenced(raw_text)
    elif mode == "cline_xml":
        content = normalize_cline_xml(raw_text)
    else:
        content = _ensure_code_fenced(raw_text)

    _last_turn_was_final = bool(
        not tool_calls and content and COMPLETION_SIGNAL_RE.search(content)
    )

    print(
        f"\033[92m[Success]\033[0m mode={mode} "
        f"text_len={len(raw_text)} tool_calls={len(tool_calls)} content_len={len(content)}"
    )

    accumulated_tokens = int(result.get("accumulatedTokens") or 0)
    initial_tokens = int(result.get("initialTokens") or 0)
    usage = usage_payload(prompt, raw_text, accumulated_tokens, initial_tokens)

    if AUTO_RESET_THRESHOLD > 0 and accumulated_tokens >= AUTO_RESET_THRESHOLD:
        print(
            f"\033[93m[Session]\033[0m Reached {accumulated_tokens} / {AUTO_RESET_THRESHOLD} tokens. Auto-resetting chat session for next turn..."
        )
        asyncio.create_task(_trigger_browser_reset())

    if not stream:
        if tool_calls:
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call["id"],
                        "type": "function",
                        "function": {
                            "name": call["name"],
                            "arguments": json.dumps(
                                call["arguments"], ensure_ascii=False
                            ),
                        },
                    }
                    for call in tool_calls
                ],
            }
            finish_reason = "tool_calls"
        else:
            message = {"role": "assistant", "content": content}
            if reasoning:
                message["reasoning_content"] = reasoning
            finish_reason = "stop"

        return web.json_response(
            {
                "id": completion_id,
                "object": "chat.completion",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": finish_reason,
                    }
                ],
                "usage": usage,
            },
            headers=CORS_HEADERS,
        )

    response = web.StreamResponse(
        status=200,
        reason="OK",
        headers={
            **CORS_HEADERS,
            "Content-Type": "text/event-stream; charset=utf-8",
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
    await response.prepare(request)

    if tool_calls:
        header = chunk(
            completion_id,
            created,
            model,
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "index": call["index"],
                        "id": call["id"],
                        "type": "function",
                        "function": {"name": call["name"], "arguments": ""},
                    }
                    for call in tool_calls
                ],
            },
        )
        print(
            f"\033[93m[Debug-Stream]\033[0m tool_calls header: {json.dumps(header, ensure_ascii=False)}"
        )
        await sse_write(response, header)

        for call in tool_calls:
            serialized = json.dumps(call["arguments"], ensure_ascii=False)
            frag_count = 0
            for fragment in iter_text_fragments(serialized):
                frag_count += 1
                arg_chunk = chunk(
                    completion_id,
                    created,
                    model,
                    {
                        "tool_calls": [
                            {
                                "index": call["index"],
                                "function": {"arguments": fragment},
                            }
                        ]
                    },
                )
                await sse_write(response, arg_chunk)
            print(
                f"\033[93m[Debug-Stream]\033[0m tool_call[{call['index']}] "
                f"name={call['name']!r} args_len={len(serialized)} frags={frag_count}"
            )

        final = chunk(completion_id, created, model, {}, "tool_calls")
        print(
            f"\033[93m[Debug-Stream]\033[0m tool_calls final: {json.dumps(final, ensure_ascii=False)}"
        )
        await sse_write(response, final)
    else:
        first = True
        frag_count = 0

        # Stream reasoning_content fragments first if thinking output is present (DeepSeek-R1)
        if reasoning:
            for fragment in iter_text_fragments(reasoning, size=64):
                delta = {"reasoning_content": fragment}
                if first:
                    delta["role"] = "assistant"
                    first = False
                await sse_write(response, chunk(completion_id, created, model, delta))

        for fragment in iter_text_fragments(content, size=64):
            frag_count += 1
            delta = {"content": fragment}
            if first:
                delta["role"] = "assistant"
                first = False
            await sse_write(response, chunk(completion_id, created, model, delta))

        print(
            f"\033[93m[Debug-Stream]\033[0m text frags={frag_count} content_len={len(content)}"
        )

        if first:
            await sse_write(
                response,
                chunk(
                    completion_id, created, model, {"role": "assistant", "content": ""}
                ),
            )

        await sse_write(response, chunk(completion_id, created, model, {}, "stop"))

    if include_usage:
        await sse_write(
            response,
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [],
                "usage": usage,
            },
        )

    await response.write(b"data: [DONE]\n\n")
    await response.write_eof()
    return response


_CLINE_ATTEMPT_RE = re.compile(
    r"<attempt_completion>\s*<result>(.*?)</result>\s*</attempt_completion>",
    re.DOTALL | re.IGNORECASE,
)
_CLINE_ASK_RE = re.compile(
    r"<ask_followup_question>\s*<question>(.*?)</question>\s*</ask_followup_question>",
    re.DOTALL | re.IGNORECASE,
)


def normalize_cline_xml(raw: str) -> str:
    text = (raw or "").strip()
    if not text:
        return (
            "<attempt_completion><result>(empty response)</result></attempt_completion>"
        )

    match = _CLINE_ATTEMPT_RE.search(text)
    if match:
        inner = match.group(1).strip()
        return (
            f"<attempt_completion>\n<result>\n{inner}\n</result>\n</attempt_completion>"
        )

    match = _CLINE_ASK_RE.search(text)
    if match:
        inner = match.group(1).strip()
        return f"<ask_followup_question>\n<question>\n{inner}\n</question>\n</ask_followup_question>"

    text = _strip_fences(text)
    if "<attempt_completion>" in text or "<ask_followup_question>" in text:
        return text

    return f"<attempt_completion>\n<result>\n{text}\n</result>\n</attempt_completion>"


@web.middleware
async def error_safety_net(request: web.Request, handler):
    #Last-resort catch-all: turns any exception this app doesn't already
    #handle gracefully (network races on browser refresh, unexpected
    #userscript payload shapes, etc.) into a proper OpenAI-style JSON error
    #instead of letting it become a raw, header-less aiohttp 500 - which is
    #what VS Code's "Sorry, your request failed... Server error: 500" comes
    #from. This is a safety net alongside (not instead of) the targeted
    #fixes in run_browser_job/stream_browser_job, for anything not already
    #anticipated there.
    try:
        return await handler(request)
    except web.HTTPException:
        raise  # deliberate responses (400s, 204 OPTIONS, etc.) pass through
    except Exception as e:
        print(f"\033[91m[Bridge]\033[0m Unhandled exception in {request.path}: {e!r}")
        import traceback

        traceback.print_exc()
        return error_response(f"Internal proxy error: {e}", 500, "server_error")


def make_app() -> web.Application:
    app = web.Application(
        client_max_size=64 * 1024 * 1024, middlewares=[error_safety_net]
    )
    app.router.add_route("POST", "/v1/chat/completions", handle_chat_completions)
    app.router.add_route("OPTIONS", "/v1/chat/completions", handle_chat_completions)
    app.router.add_route("GET", "/v1/models", handle_models)
    app.router.add_route("OPTIONS", "/v1/models", handle_models)
    app.router.add_get("/health", handle_health)
    app.router.add_get("/ws", websocket_handler)
    return app


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="OpenAI-compatible proxy for DeepSeek Web"
    )
    parser.add_argument("--host", default=HOST, help=f"Host to bind (default: {HOST})")
    parser.add_argument(
        "--port", type=int, default=PORT, help=f"Port to bind (default: {PORT})"
    )
    parser.add_argument(
        "--reset-threshold",
        type=int,
        default=AUTO_RESET_THRESHOLD,
        help=f"Auto-reset chat session token threshold (default: {AUTO_RESET_THRESHOLD}, 0 to disable)",
    )
    args = parser.parse_args()

    print(
        f"\033[95m[Boot]\033[0m OpenAI-compatible bridge on http://{args.host}:{args.port}/v1"
    )
    print(f"\033[95m[Boot]\033[0m WebSocket endpoint:  ws://{args.host}:{args.port}/ws")
    AUTO_RESET_THRESHOLD = args.reset_threshold
    web.run_app(make_app(), host=args.host, port=args.port, print=None)