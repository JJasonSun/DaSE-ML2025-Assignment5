from importlib import import_module
from typing import Dict

from core.agent_factory import parse_agent_spec


DEFAULT_AGENT_PROFILE: Dict[str, str] = {
    "positioning": "Custom or unknown agent. Only the agent path is available in the structured data.",
    "expected_strengths": "Do not infer strengths that are not visible from metrics, traces, or repository documentation.",
    "expected_limits": "Do not infer implementation-specific limits unless they are supported by provided trace fields.",
    "analysis_focus": "Ground the analysis in explicit metrics, bad cases, and available trace metadata.",
}


def agent_profile_for(agent_spec: str) -> Dict[str, str]:
    module_name, class_name = _parse_agent_spec_or_unknown(agent_spec)
    profile = _profile_from_agent_class(module_name, class_name)
    return {"name": class_name or "unknown", **profile}


def _parse_agent_spec_or_unknown(agent_spec: str) -> tuple[str, str]:
    try:
        return parse_agent_spec(agent_spec)
    except Exception:
        class_name = (agent_spec or "unknown").split(":")[-1].split(".")[-1]
        return "", class_name or "unknown"


def _profile_from_agent_class(module_name: str, class_name: str) -> Dict[str, str]:
    if not module_name or not class_name:
        return dict(DEFAULT_AGENT_PROFILE)

    try:
        agent_class = getattr(import_module(module_name), class_name)
    except Exception:
        return dict(DEFAULT_AGENT_PROFILE)

    raw_profile = getattr(agent_class, "AGENT_PROFILE", None)
    if not isinstance(raw_profile, dict):
        return dict(DEFAULT_AGENT_PROFILE)

    profile = dict(DEFAULT_AGENT_PROFILE)
    for key in ("positioning", "expected_strengths", "expected_limits", "analysis_focus"):
        value = raw_profile.get(key)
        if isinstance(value, str) and value.strip():
            profile[key] = value.strip()
    return profile
