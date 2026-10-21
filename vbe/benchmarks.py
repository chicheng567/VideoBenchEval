"""Loaders that turn each benchmark's annotation file into one flat list of
multiple-choice samples:

    {"qid", "video", "question", "options": [str, ...], "answer": "A".."E", "meta": {...}}

`video` is an absolute path. `meta` holds the fields the scorer breaks results down by.
"""
import ast
import json
import os
import re
from typing import Dict, List

import pandas as pd

LETTERS = "ABCDEFGH"
DEFAULT_ROOT = "/workspace/chicheng/datasets"


def _videomme(root: str) -> List[Dict]:
    d = os.path.join(root, "videomme")
    df = pd.read_parquet(os.path.join(d, "videomme", "test-00000-of-00001.parquet"))
    out = []
    for r in df.itertuples():
        # options are already "A. xxx"; strip the letter so every benchmark is formatted the same way
        opts = [re.sub(r"^[A-H]\.\s*", "", o) for o in r.options]
        out.append(dict(
            qid=r.question_id,
            video=os.path.join(d, "data", f"{r.videoID}.mp4"),
            question=r.question,
            options=opts,
            answer=r.answer.strip(),
            meta=dict(duration=r.duration, domain=r.domain, task_type=r.task_type),
        ))
    return out


def _mlvu(root: str) -> List[Dict]:
    d = os.path.join(root, "mlvu")
    df = pd.read_parquet(os.path.join(d, "mlvu", "test-00000-of-00001.parquet"))
    out = []
    for r in df.itertuples():
        # the question text embeds "(A) ..." option lines; keep only the question itself
        q = re.split(r"\n\(A\)", r.question, maxsplit=1)[0].strip()
        out.append(dict(
            qid=f"{r.task_type}/{r.question_id}",
            video=os.path.join(d, r.video_name),
            question=q,
            options=list(r.candidates),
            answer=r.answer.strip(),
            meta=dict(task_type=r.task_type, duration=float(r.duration)),
        ))
    return out


def _longvideobench(root: str) -> List[Dict]:
    d = os.path.join(root, "longvideobench")
    out = []
    for r in json.load(open(os.path.join(d, "lvb_val.json"))):
        cands = r["candidates"]
        if isinstance(cands, str):
            cands = ast.literal_eval(cands)
        out.append(dict(
            qid=r["id"],
            video=os.path.join(d, "videos", r["video_path"]),
            question=r["question"],
            options=list(cands),
            answer=LETTERS[int(r["correct_choice"])],
            meta=dict(duration_group=int(r["duration_group"]), question_category=r["question_category"],
                      level=r["level"]),
        ))
    return out


BENCHMARKS = {
    "videomme": _videomme,
    "mlvu": _mlvu,
    "longvideobench": _longvideobench,
}


def load_benchmark(name: str, root: str = DEFAULT_ROOT) -> List[Dict]:
    samples = BENCHMARKS[name](root)
    for s in samples:
        assert 2 <= len(s["options"]) <= len(LETTERS), s["qid"]
        assert s["answer"] in LETTERS[: len(s["options"])], (s["qid"], s["answer"])
    return samples


# Prompts and answer parsing follow the official VideoLLaMA3 evaluation
# (VideoLLaMA3/evaluation/benchmarks/{videomme,mlvu,longvideobench}.py), without subtitles.

def build_prompt(bench: str, sample: Dict) -> str:
    q, opts = sample["question"], sample["options"]
    letters = LETTERS[: len(opts)]
    if bench == "videomme":
        return (
            "Select the best answer to the following multiple-choice question based on the video. "
            "Respond with only the letter (A, B, C, or D) of the correct option.\n"
            f"{q}\n" + "".join(f"{l}. {o}\n" for l, o in zip(letters, opts))
            + "Answer with the option's letter from the given choices directly and only give the best option. "
            "The best answer is:"
        )
    if bench == "mlvu":
        return (
            f"Question: {q}\nOptions:\n" + "".join(f"({l}) {o}\n" for l, o in zip(letters, opts))
            + "Answer with the option's letter from the given choices directly and only give the best option."
        )
    if bench == "longvideobench":
        return "\n".join([f"Question: {q}"] + [f"{l}. {o}" for l, o in zip(letters, opts)]
                         + ["Answer with the option's letter from the given choices directly."])
    raise KeyError(bench)


def extract_choice(bench: str, response: str, options: List[str]) -> str:
    """Official parsing: first option letter (bracketed / spaced) after dropping the word
    'answer'; otherwise the first option whose text appears in the response; '' if neither."""
    letters = LETTERS[: len(options)]
    r = response.replace("answer", "").replace("Answer", "")
    if bench == "mlvu":
        m = re.findall(rf"[\(,\ ]*[{letters[0]}-{letters[-1]}][\),\ ]*", r)
        if m:
            return re.search(rf"[{letters[0]}-{letters[-1]}]", m[0]).group(0)
    else:
        m = re.findall(rf"[\(\ \[]*([{letters[0]}-{letters[-1]}])[\)\.\ \]]*", r)
        if m:
            return m[0]
    digit2word = dict(zip("0123456789", ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]))
    for i, opt in enumerate(options):
        opt = opt.strip().strip(".")
        alt = digit2word.get(opt, opt)
        if opt and (opt.lower() in response.lower() or alt.lower() in response.lower()):
            return letters[i]
    return ""
