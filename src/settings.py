"""Study paths, the agent model and `.env` loading."""
import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
BENCHMARKS = WORKSPACE / "benchmarks"
RESULTS = PROJECT / "results"
# The upstream MedAgentBench FHIR server image, pinned by digest.
FHIR_IMAGE = "jyxsu6/medagentbench@sha256:3fb83d7ed71c5476f9eb6212bd440a909ef7505922bbc757dc488a8fc0701966"

# The agent models, all called at OpenRouter (`openrouter_transport.py`). Each call is pinned to the model's own
# provider (`provider`, no fallbacks); the base slug "openai" is the standard tier, without the Flex and Fast
# endpoints. The cost of each call is the one that OpenRouter reports. Each run records its model entry in config.json.
MODELS = {
    "gpt-6-luna": {"name": "gpt-6-luna", "openrouter_id": "openai/gpt-6-luna", "provider": "openai",
                   "service_tier": None},
    "gpt-6.1-terra": {"name": "gpt-6.1-terra", "openrouter_id": "openai/gpt-6.1-terra", "provider": "openai",
                      "service_tier": None},
    "claude-haiku-5-5": {"name": "claude-haiku-5-5", "openrouter_id": "anthropic/claude-haiku-5.5",
                         "provider": "anthropic", "service_tier": None},
    "claude-sonnet-5-5": {"name": "claude-sonnet-5-5", "openrouter_id": "anthropic/claude-sonnet-5.5",
                          "provider": "anthropic", "service_tier": None},
}
MODEL = MODELS["gpt-6-luna"]


def load_env_file(path: Path | None = None) -> None:
    """Load simple KEY=VALUE lines without replacing existing environment values."""
    candidates = [path] if path else [PROJECT / ".env", Path.cwd() / ".env"]
    for candidate in candidates:
        if candidate is None or not candidate.is_file():
            continue
        try:
            lines = candidate.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError):
            continue
        for line in lines:
            text = line.strip()
            if not text or text.startswith("#") or "=" not in text:
                continue
            name, value = (part.strip() for part in text.split("=", 1))
            if name.startswith("export "):
                name = name[7:].strip()
            if not name or not (name.replace("_", "").isalnum()):
                continue
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            os.environ.setdefault(name, value)
        return
