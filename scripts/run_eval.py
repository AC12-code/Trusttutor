"""Run the grounding eval from the command line.

Examples:
  python -m scripts.run_eval
  python -m scripts.run_eval --transcript data/sample_transcript.json
  python -m scripts.run_eval --youtube https://youtu.be/VIDEO_ID
  LLM_PROVIDER=mock python -m scripts.run_eval        # offline wiring test
"""
import argparse
from app.ingest import load_from_file, load_from_youtube
from app.retrieve import chunk_segments, Retriever
from app.tutor import Tutor
from app import eval_harness


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcript", default="data/sample_transcript.json")
    ap.add_argument("--youtube", default=None)
    ap.add_argument("--questions", default="data/eval_questions.json")
    ap.add_argument("--report", default="report.html")
    args = ap.parse_args()

    segs = load_from_youtube(args.youtube) if args.youtube else load_from_file(args.transcript)
    print(f"Loaded {len(segs)} segments.")
    retriever = Retriever(chunk_segments(segs))
    print(f"Built {len(retriever.chunks)} chunks via '{retriever.backend}' retriever.")
    tutor = Tutor(retriever)

    questions = eval_harness.load_questions(args.questions)
    rep = eval_harness.run(tutor, questions)
    eval_harness.print_summary(rep)
    eval_harness.write_html(rep, args.report)
    print(f"\nWrote {args.report}")


if __name__ == "__main__":
    main()
