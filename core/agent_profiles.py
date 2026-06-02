from typing import Dict


AGENT_PROFILES: Dict[str, Dict[str, str]] = {
    "BaselineAgent": {
        "positioning": "Minimal control group. It uses a simple context window and does not perform hybrid retrieval or deterministic tool execution.",
        "expected_strengths": "Useful as a lower-bound baseline for checking whether the evaluation pipeline and prompt format are working.",
        "expected_limits": "Likely to miss deeply buried evidence, multi-document evidence joins, and exact computation-heavy tasks.",
        "analysis_focus": "Treat failures mainly as baseline capability gaps unless metrics show answer formatting or evaluator issues.",
    },
    "HybridRetrievalAgent": {
        "positioning": "Retrieval-augmented baseline using BM25, dense embeddings, rerank, and neighbor chunk expansion before final LLM answering.",
        "expected_strengths": "Designed for evidence recall across larger multi-document contexts and more stable retrieval than a simple baseline.",
        "expected_limits": "Still relies on the LLM for exact arithmetic, date reasoning, string operations, encoding, and final answer formatting.",
        "analysis_focus": "Separate retrieval failures from post-retrieval reasoning or formatting failures. Suggestions should target retrieval, rerank, chunking, and final-answer prompting.",
    },
    "ToolAugmentedAgent": {
        "positioning": "Productized agent combining hybrid retrieval, structured evidence extraction, deterministic Python tools, fallback handling, and trace metadata.",
        "expected_strengths": "Designed to improve exact arithmetic, date reasoning, string analysis, encoding/decoding, and traceable failure attribution.",
        "expected_limits": "Quality depends on task classification, evidence extraction completeness, operation parsing, tool execution coverage, and fallback behavior.",
        "analysis_focus": "Use tool diagnostics heavily. Attribute failures to classification, evidence extraction, tool execution, fallback-to-hybrid, or answer normalization only when supported by traces and metrics.",
    },
}


def agent_profile_for(agent_spec: str) -> Dict[str, str]:
    class_name = (agent_spec or "unknown").split(":")[-1].split(".")[-1]
    profile = AGENT_PROFILES.get(class_name)
    if profile:
        return {"name": class_name, **profile}
    return {
        "name": class_name or "unknown",
        "positioning": "Custom or unknown agent. Only the agent path is available in the structured data.",
        "expected_strengths": "Do not infer strengths that are not visible from metrics, traces, or repository documentation.",
        "expected_limits": "Do not infer implementation-specific limits unless they are supported by provided trace fields.",
        "analysis_focus": "Ground the analysis in explicit metrics, bad cases, and available trace metadata.",
    }
