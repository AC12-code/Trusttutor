"""The money shot: measure the hallucination gap.

We run each eval question through BOTH the grounded tutor and the naive LLM.
Questions are labelled `answerable` (the lesson covers it) or not (adversarial /
off-topic). Then we score behaviour:

  * hallucination_rate  : share of UNANSWERABLE questions the model still answered
                          (grounded should refuse; naive typically won't).
  * refusal_accuracy    : share of UNANSWERABLE questions correctly refused.
  * citation_coverage   : share of ANSWERABLE answers that carry a timestamp cite.

Factual correctness needs a human eye, but refusal + citation behaviour is the
trust signal a founder cares about, and it's fully measurable offline.
"""
from __future__ import annotations
import html
import json
from dataclasses import dataclass, field
from typing import List
from .tutor import Tutor, naive_answer


@dataclass
class Row:
    question: str
    answerable: bool
    grounded_refused: bool
    naive_refused: bool
    grounded_cited: bool
    grounded_answer: str
    naive_answer: str


@dataclass
class Report:
    rows: List[Row] = field(default_factory=list)

    def metrics(self) -> dict:
        unans = [r for r in self.rows if not r.answerable]
        ans = [r for r in self.rows if r.answerable]
        def rate(xs):  # guard div-by-zero
            return round(100 * xs, 1)
        naive_halluc = (sum(1 for r in unans if not r.naive_refused) / len(unans)) if unans else 0
        grounded_halluc = (sum(1 for r in unans if not r.grounded_refused) / len(unans)) if unans else 0
        refusal_acc = (sum(1 for r in unans if r.grounded_refused) / len(unans)) if unans else 0
        cite_cov = (sum(1 for r in ans if r.grounded_cited) / len(ans)) if ans else 0
        return {
            "naive_hallucination_rate_pct": rate(naive_halluc),
            "grounded_hallucination_rate_pct": rate(grounded_halluc),
            "grounded_refusal_accuracy_pct": rate(refusal_acc),
            "grounded_citation_coverage_pct": rate(cite_cov),
            "n_questions": len(self.rows),
            "n_unanswerable": len(unans),
        }


def _looks_refused(a: dict) -> bool:
    return a.get("refused") or "can't answer" in a.get("answer", "").lower()


def _has_citation(a: dict) -> bool:
    return bool(a.get("citations")) or "[" in a.get("answer", "")


def run(tutor: Tutor, questions: List[dict]) -> Report:
    rep = Report()
    for q in questions:
        g = tutor.ask(q["q"])
        n = naive_answer(q["q"])
        rep.rows.append(Row(
            question=q["q"],
            answerable=bool(q["answerable"]),
            grounded_refused=_looks_refused(g),
            naive_refused=_looks_refused(n),
            grounded_cited=_has_citation(g),
            grounded_answer=g["answer"],
            naive_answer=n["answer"],
        ))
    return rep


def load_questions(path: str) -> List[dict]:
    return json.load(open(path))


def print_summary(rep: Report) -> None:
    m = rep.metrics()
    print("\n=== TrustTutor eval ===")
    print(f"Questions: {m['n_questions']}  (unanswerable/adversarial: {m['n_unanswerable']})")
    print(f"Naive LLM hallucination rate    : {m['naive_hallucination_rate_pct']}%")
    print(f"Grounded tutor hallucination    : {m['grounded_hallucination_rate_pct']}%")
    print(f"Grounded refusal accuracy       : {m['grounded_refusal_accuracy_pct']}%")
    print(f"Grounded citation coverage      : {m['grounded_citation_coverage_pct']}%")
    print("-" * 40)
    for r in rep.rows:
        tag = "ANS " if r.answerable else "ADV "
        gi = "refused" if r.grounded_refused else "answered"
        ni = "refused" if r.naive_refused else "answered"
        print(f"[{tag}] grounded={gi:8s} naive={ni:8s} | {r.question[:60]}")


def write_html(rep: Report, path: str) -> None:
    m = rep.metrics()
    def cell(ok, txt):
        color = "#1a7f37" if ok else "#cf222e"
        return f'<td style="color:{color};font-weight:600">{html.escape(txt)}</td>'
    rows = []
    for r in rep.rows:
        good_g = r.grounded_refused if not r.answerable else (not r.grounded_refused)
        bad_n = (not r.naive_refused) and (not r.answerable)
        rows.append(
            "<tr>"
            f"<td>{'adversarial' if not r.answerable else 'answerable'}</td>"
            f"<td>{html.escape(r.question)}</td>"
            + cell(good_g, "refused" if r.grounded_refused else "answered")
            + cell(not bad_n, "refused" if r.naive_refused else "answered")
            + "</tr>"
            '<tr><td colspan="4" style="color:#656d76;font-size:12px;padding:0 0 12px 0">'
            f"<b>grounded:</b> {html.escape(r.grounded_answer)}<br>"
            f"<b>naive:</b> {html.escape(r.naive_answer)}</td></tr>"
        )
    doc = f"""<!doctype html><meta charset="utf-8">
<title>TrustTutor eval</title>
<body style="font-family:system-ui,-apple-system,sans-serif;max-width:900px;margin:40px auto;color:#1f2328">
<h1>TrustTutor — grounding eval</h1>
<div style="display:flex;gap:16px;flex-wrap:wrap;margin:20px 0">
  <div style="padding:16px;border:1px solid #d0d7de;border-radius:10px">
    <div style="font-size:12px;color:#656d76">NAIVE LLM HALLUCINATION</div>
    <div style="font-size:32px;font-weight:700;color:#cf222e">{m['naive_hallucination_rate_pct']}%</div></div>
  <div style="padding:16px;border:1px solid #d0d7de;border-radius:10px">
    <div style="font-size:12px;color:#656d76">GROUNDED HALLUCINATION</div>
    <div style="font-size:32px;font-weight:700;color:#1a7f37">{m['grounded_hallucination_rate_pct']}%</div></div>
  <div style="padding:16px;border:1px solid #d0d7de;border-radius:10px">
    <div style="font-size:12px;color:#656d76">REFUSAL ACCURACY</div>
    <div style="font-size:32px;font-weight:700">{m['grounded_refusal_accuracy_pct']}%</div></div>
  <div style="padding:16px;border:1px solid #d0d7de;border-radius:10px">
    <div style="font-size:12px;color:#656d76">CITATION COVERAGE</div>
    <div style="font-size:32px;font-weight:700">{m['grounded_citation_coverage_pct']}%</div></div>
</div>
<table style="border-collapse:collapse;width:100%">
<thead><tr style="text-align:left;border-bottom:2px solid #d0d7de">
<th>type</th><th>question</th><th>grounded</th><th>naive</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<p style="color:#656d76;font-size:13px;margin-top:20px">Green = desired behaviour.
Grounded tutor should refuse adversarial questions and answer covered ones; the
naive baseline answers everything, including what it can't know.</p>
</body>"""
    open(path, "w", encoding="utf-8").write(doc)
