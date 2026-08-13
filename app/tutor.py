"""The grounded tutor, as a small but real LangGraph state machine.

Flow:  retrieve -> grade_grounding -> (answer_with_citations | refuse)

The whole point: the tutor answers ONLY from the lesson and cites timestamps,
and refuses when the material doesn't cover the question. `naive_answer` is the
ungrounded baseline (plain LLM, no context, no refusal) used by the eval harness
to expose the hallucination gap.
"""
from __future__ import annotations
from typing import List, Optional, TypedDict
from .config import settings
from .retrieve import Retriever, Chunk
from .llm import complete

REFUSAL = "I can't answer that from this lesson — it isn't covered in the material."
REFUSAL_TOKEN = "NOT_IN_LESSON"

ANSWER_SYSTEM = (
    "You are a careful tutor. Answer the student's question USING ONLY the "
    "provided lesson context. Cite the timestamp(s) you used in square brackets, "
    "e.g. [02:15]. If the context does not contain the answer, reply with exactly "
    f"'{REFUSAL_TOKEN}' and nothing else. Never use outside knowledge."
)

GRADER_SYSTEM = (
    "You are grading whether a lesson context is sufficient to answer a question. "
    "Reply with exactly 'yes' or 'no'."
)


class TutorState(TypedDict, total=False):
    question: str
    hits: List[tuple]          # (Chunk, score)
    grounded: bool
    answer: str
    citations: List[str]


class Tutor:
    def __init__(self, retriever: Retriever):
        self.retriever = retriever
        self.graph = self._build_graph()

    # --- nodes ---
    def _retrieve(self, state: TutorState) -> TutorState:
        hits = self.retriever.search(state["question"])
        return {"hits": hits}

    def _grade(self, state: TutorState) -> TutorState:
        hits = state.get("hits", [])
        best = hits[0][1] if hits else 0.0
        grounded = best >= settings.grounding_threshold
        # Optional stricter second gate: let the model veto weak-but-present context.
        if grounded and settings.use_llm_grader:
            ctx = "\n".join(c.text for c, _ in hits)
            verdict = complete(GRADER_SYSTEM,
                               f"Question: {state['question']}\nContext: {ctx}").lower()
            grounded = verdict.strip().startswith("yes")
        return {"grounded": grounded}

    def _answer(self, state: TutorState) -> TutorState:
        hits = state["hits"]
        ctx = "\n".join(f"[{c.ts()}] {c.text}" for c, _ in hits)
        out = complete(ANSWER_SYSTEM, f"Question: {state['question']}\nContext: {ctx}")
        if REFUSAL_TOKEN in out:
            return {"answer": REFUSAL, "grounded": False, "citations": []}
        cites = [c.ts() for c, _ in hits]
        return {"answer": out, "citations": cites}

    def _refuse(self, state: TutorState) -> TutorState:
        return {"answer": REFUSAL, "citations": []}

    def _route(self, state: TutorState) -> str:
        return "answer" if state.get("grounded") else "refuse"

    def _build_graph(self):
        try:
            from langgraph.graph import StateGraph, END
        except ImportError as e:
            raise RuntimeError("pip install langgraph") from e
        g = StateGraph(TutorState)
        g.add_node("retrieve", self._retrieve)
        g.add_node("grade", self._grade)
        g.add_node("answer", self._answer)
        g.add_node("refuse", self._refuse)
        g.set_entry_point("retrieve")
        g.add_edge("retrieve", "grade")
        g.add_conditional_edges("grade", self._route, {"answer": "answer", "refuse": "refuse"})
        g.add_edge("answer", END)
        g.add_edge("refuse", END)
        return g.compile()

    def ask(self, question: str) -> dict:
        final = self.graph.invoke({"question": question})
        return {
            "question": question,
            "answer": final.get("answer", REFUSAL),
            "grounded": final.get("grounded", False),
            "citations": final.get("citations", []),
            "refused": final.get("answer") == REFUSAL,
        }


def naive_answer(question: str) -> dict:
    """Ungrounded baseline: no retrieval, no refusal path, no citations."""
    out = complete(
        "You are a helpful tutor. Answer the question.",
        f"Question: {question}",
    )
    return {"question": question, "answer": out, "grounded": False,
            "citations": [], "refused": False}
