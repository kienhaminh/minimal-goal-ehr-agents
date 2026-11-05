"""OpenRouter Chat Completions transport of the agent models, authenticated by OPENROUTER_API_KEY.

The client keeps the call shape of the OpenAI Responses SDK, so the agent loop of the upstream benchmark serves it:
`responses.create(**kw)` takes the Responses arguments of `agent.run` and returns a Responses-shaped result.
- `model`: the OpenRouter id of the model entry (`settings.MODELS`, `openrouter_id`).
- `input`: system, user and assistant messages keep their text; a function_call becomes an assistant message with one
  tool call, and a function_call_output a tool message. Each returned function_call carries the full assistant message
  it came from (`chat_message`, with its reasoning details), which replaces the call when the input is rebuilt, so the
  model gets its reasoning back with its tool call.
- `reasoning.effort` -> `reasoning.effort`; `max_output_tokens` -> `max_tokens`; `parallel_tool_calls` is passed on; the
  service tier is not sent.
- The provider is pinned (`provider.only`, the `provider` of the model entry, no fallbacks), so every call of a run
  reaches the same deployment. OpenRouter lists no `parallel_tool_calls` for these models, so the calls are sent
  without `require_parameters`. Should a response hold several tool calls when the agent asks for one per
  step, the first is kept and `usage.dropped_tool_calls` counts the others.
- Output: the text becomes one message item and each tool call a function_call item. `status` is the finish reason.

The key is read from the environment or the .env file of the project folder and used only as the bearer credential of
https://openrouter.ai/api/v1/chat/completions. No retries. Each call records `usage.route = "openrouter"`, and
`usage.cost` is the cost that OpenRouter reports for the call; `input_tokens`, `output_tokens` and their details follow
the Responses usage names.
"""
import json
import os
from urllib.error import HTTPError
from urllib.request import Request, build_opener

from clinical_agent.settings import load_env_file

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"


class TransportError(RuntimeError):
    """A request without a usable reply; the message holds codes only, never bodies or credentials."""


def chat_messages(items):
    """Chat Completions messages for the Responses input items of `agent.run`."""
    messages, seen = [], set()
    for i, item in enumerate(items):
        if item.get("type") == "function_call":
            if id(item["chat_message"]) not in seen:  # one assistant message may carry several calls
                seen.add(id(item["chat_message"]))
                messages.append(item["chat_message"])
        elif item.get("type") == "function_call_output":
            messages.append({"role": "tool", "tool_call_id": item["call_id"], "content": item["output"]})
        elif item["role"] == "assistant":
            following = items[i + 1] if i + 1 < len(items) else {}
            if following.get("type") != "function_call":  # else chat_message holds the text
                messages.append({"role": "assistant", "content": item["content"]})
        else:
            messages.append({"role": item["role"], "content": item["content"]})
    return messages


def tool_of(schema):
    """A Responses function tool as a Chat Completions tool."""
    return {"type": "function", "function": {"name": schema["name"], "description": schema.get("description", ""),
                                             "parameters": schema["parameters"]}}


class Response:
    def __init__(self, data):
        self.data = data
        self.output = data["output"]

    def model_dump(self, mode=None):
        return self.data


class OpenRouterChat:
    def __init__(self, model, timeout=300):
        self.model, self.timeout = model, timeout
        self.responses = self  # call shape of the OpenAI SDK: client.responses.create(**kw)
        self.credential()  # fail before the first attempt, not inside it

    def close(self):
        pass

    def credential(self):
        load_env_file()
        key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if not key:
            raise TransportError("Missing OPENROUTER_API_KEY in the environment or the .env file of the project folder")
        return key

    def create(self, *, model, input, tools, parallel_tool_calls, reasoning, max_output_tokens, service_tier=None):
        body = {"model": self.model["openrouter_id"], "messages": chat_messages(input),
                "tools": [tool_of(t) for t in tools], "parallel_tool_calls": parallel_tool_calls,
                "reasoning": {"effort": reasoning["effort"]}, "max_tokens": max_output_tokens, "usage": {"include": True},
                "provider": {"only": [self.model["provider"]], "allow_fallbacks": False, "require_parameters": False}}
        request = Request(ENDPOINT, json.dumps(body).encode(), {"Authorization": "Bearer " + self.credential(),
                                                                "Content-Type": "application/json"})
        try:
            with build_opener().open(request, timeout=self.timeout) as reply:
                data = json.loads(reply.read())
        except HTTPError as error:
            try:
                detail = json.loads(error.read(65536)).get("error") or {}
            except (ValueError, AttributeError):
                detail = {}
            # code only; the provider message is not kept
            raise TransportError("HTTP %d code=%s" % (error.code, detail.get("code"))) from None
        if data.get("error") or not data.get("choices"):
            raise TransportError("No choices, error code=%s" % (data.get("error") or {}).get("code"))
        choice = data["choices"][0]
        message = {k: v for k, v in choice["message"].items() if k not in ("refusal", "annotations")}
        message["role"] = "assistant"
        output = []
        if message.get("content"):
            output.append({"type": "message", "role": "assistant",
                           "content": [{"type": "output_text", "text": message["content"]}]})
        output += [{"type": "function_call", "call_id": c["id"], "name": c["function"]["name"],
                    "arguments": c["function"]["arguments"] or "{}", "chat_message": message}
                   for c in message.get("tool_calls") or []]
        usage = data.get("usage") or {}
        normalized = {"input_tokens": usage.get("prompt_tokens"), "output_tokens": usage.get("completion_tokens"),
                      "input_tokens_details": {"cached_tokens": (usage.get("prompt_tokens_details") or {})
                                               .get("cached_tokens", 0)},
                      "output_tokens_details": {"reasoning_tokens": (usage.get("completion_tokens_details") or {})
                                                .get("reasoning_tokens", 0)},
                      "raw": usage, "cost": usage.get("cost"), "provider": data.get("provider")}
        calls = [o for o in output if o["type"] == "function_call"]
        dropped = len(calls) - 1 if not parallel_tool_calls and len(calls) > 1 else 0
        if dropped:
            first = calls[0]
            message = {**first["chat_message"], "tool_calls": first["chat_message"]["tool_calls"][:1]}
            output = [o for o in output if o["type"] != "function_call"] + [{**first, "chat_message": message}]
        normalized |= {"route": "openrouter", "dropped_tool_calls": dropped}
        return Response({"model": data.get("model"), "service_tier": None, "status": choice.get("finish_reason"),
                         "usage": normalized, "output": output})
