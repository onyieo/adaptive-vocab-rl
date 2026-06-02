"""Generate a ~500-word JLPT N5/N4/N3 vocabulary list via Claude.

Run once. Calls Claude in batches (rate-limit friendly), validates the output
(unique surfaces, plausible readings), and saves to vocab.json. simulator.py
prefers this file over the embedded 51-word default if it exists.

The decision to use an LLM to generate the list is deliberate: hand-writing
500 correct (surface, reading, gloss, level) tuples is error-prone, and we
can validate with simple structural checks. Treat the generated list as a
research artifact — re-run if you want a different sample.

Usage:
    python generate_vocab.py             # default: 500 words
    python generate_vocab.py --n 1000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import anthropic

from llm._env import require_api_key


MODEL = "claude-sonnet-4-6"
BATCH_SIZE = 100  # words per API call
LEVEL_SPLIT = {"N5": 0.35, "N4": 0.35, "N3": 0.30}


SYSTEM = """You generate JLPT vocabulary lists for a language-learning research project.

Output requirements:
- Each entry must be a real, common Japanese word at the requested JLPT level.
- "surface" is the word as it appears in writing (kanji where natural, kana otherwise — e.g. 水, たべる, 大きい). Use the most common form.
- "reading" is the full kana reading (hiragana for native words, katakana for loanwords).
- "gloss" is a brief English meaning (1-4 words, lowercase, no period).
- Do NOT include particles, suffixes, or conjugated forms.
- Do NOT repeat words from prior batches in this conversation.
- Mix common parts of speech proportionally (nouns ~50%, verbs ~25%, adjectives ~15%, others ~10%).
- For verbs, use the dictionary form (e.g. 食べる, not 食べます).
- For i-adjectives, include the trailing い (e.g. 大きい).
- For na-adjectives, omit な (e.g. 静か, not 静かな).
"""


SCHEMA = {
    "type": "object",
    "properties": {
        "words": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "surface": {"type": "string"},
                    "reading": {"type": "string"},
                    "gloss":   {"type": "string"},
                    "jlpt":    {"type": "string", "enum": ["N5", "N4", "N3"]},
                },
                "required": ["surface", "reading", "gloss", "jlpt"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["words"],
    "additionalProperties": False,
}


def _generate_batch(client, level: str, n: int,
                    avoid: set[str]) -> list[dict]:
    avoid_chunk = ""
    if avoid:
        sample = list(avoid)[-200:]  # last 200 to keep prompt small
        avoid_chunk = "\nAvoid these (already in earlier batches): " + ", ".join(sample)
    user = (f"Generate {n} JLPT {level} vocabulary entries. Output as JSON with the "
            f"requested schema.{avoid_chunk}")

    response = client.messages.create(
        model=MODEL,
        max_tokens=12_000,
        system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user}],
        output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
    )
    text = "".join(b.text for b in response.content if b.type == "text")
    parsed = json.loads(text)
    return parsed["words"]


def _validate(entries: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    skipped = Counter()
    for e in entries:
        s = e["surface"].strip()
        r = e["reading"].strip()
        g = e["gloss"].strip().lower()
        lvl = e["jlpt"].strip()

        if s in seen:
            skipped["duplicate_surface"] += 1; continue
        if not s or not r or not g:
            skipped["empty_field"] += 1; continue
        if lvl not in ("N5", "N4", "N3"):
            skipped["bad_level"] += 1; continue
        # cheap sanity: reading should be mostly kana
        kana_chars = sum(1 for c in r
                         if "぀" <= c <= "ゟ" or "゠" <= c <= "ヿ")
        if kana_chars < len(r) * 0.7:
            skipped["bad_reading"] += 1; continue
        seen.add(s)
        out.append({"surface": s, "reading": r, "gloss": g, "jlpt": lvl})

    if skipped:
        print(f"  skipped: {dict(skipped)}")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--out", default="vocab.json")
    args = p.parse_args()

    targets = {lvl: max(1, round(args.n * frac))
               for lvl, frac in LEVEL_SPLIT.items()}
    print(f"target: {targets}")

    require_api_key()
    client = anthropic.Anthropic()
    all_entries: list[dict] = []
    avoid: set[str] = set()

    for level, target_n in targets.items():
        print(f"\nLevel {level} ({target_n} words)...")
        produced = 0
        while produced < target_n:
            batch_size = min(BATCH_SIZE, target_n - produced + 30)  # slight overshoot for dropouts
            try:
                batch = _generate_batch(client, level, batch_size, avoid)
            except Exception as e:
                print(f"  batch failed: {e}; retrying with smaller batch")
                batch = _generate_batch(client, level, batch_size // 2, avoid)
            valid = _validate(batch)
            new = [w for w in valid if w["surface"] not in avoid
                   and w["jlpt"] == level]
            for w in new:
                avoid.add(w["surface"])
            all_entries.extend(new)
            produced += len(new)
            print(f"  +{len(new)} (running: {produced}/{target_n} {level})")

    print(f"\nTotal: {len(all_entries)} entries")
    counts = Counter(e["jlpt"] for e in all_entries)
    print(f"  by level: {dict(counts)}")

    here = os.path.dirname(os.path.abspath(__file__))
    out_path = args.out if os.path.isabs(args.out) else os.path.join(here, args.out)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_entries, f, indent=2, ensure_ascii=False)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
