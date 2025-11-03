from typing import Dict, Any, Generator
from langchain_community.llms import Ollama
from .config import OLLAMA_MODEL, OLLAMA_HOST, TOP_K

SYSTEM_PROMPT = """You are a manufacturing support assistant for test/repair lines.
You receive:
- User query describing symptoms (Failure Code/Description/Defect/Details and/or Part/Type/Station).
- Aggregated results from historical logs (recent-first for repeated serials).
Your goals:
1) Propose the most likely remediation steps (Action Codes) based on similar cases.
2) List the most used Materials (Material Code + Description + PartClass) seen with those issues.
3) Flag recurring serial numbers when present.
4) Be concise, accurate, and avoid speculation or fabricating codes."""

def stream_response(query: str, agg: Dict[str, Any]) -> Generator[str, None, None]:
    prompt = f"{SYSTEM_PROMPT}\n\nUser query:\n{query}\n\nContext:\n{agg}"
    llm = Ollama(model=OLLAMA_MODEL, base_url=OLLAMA_HOST)
    for token in llm.stream(prompt):
        yield token