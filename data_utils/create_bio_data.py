import argparse
import json
import random
from pathlib import Path
from typing import List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a labeled bio dataset JSON by sampling questions from "
            "remove/retain WMDP JSONL files."
        )
    )
    parser.add_argument(
        "--remove-path",
        type=Path,
        default=Path(
            "/home/morg/students/rashkovits/Localized-UNDO/datasets/wmdp/qa/"
            "wmdp-bio_remove_dataset-combined.jsonl"
        ),
        help="Path to remove split JSONL (mapped to label 'bio_forget').",
    )
    parser.add_argument(
        "--retain-path",
        type=Path,
        default=Path(
            "/home/morg/students/rashkovits/Localized-UNDO/datasets/wmdp/qa/"
            "wmdp-bio_retain_dataset-combined.jsonl"
        ),
        help="Path to retain split JSONL (mapped to label 'neutral').",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/bio_data.json"),
        help="Output JSON path.",
    )
    parser.add_argument(
        "--samples-per-label",
        type=int,
        default=500,
        help="How many samples to draw for each label.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for reproducible sampling.",
    )
    return parser.parse_args()


def load_questions(jsonl_path: Path) -> List[str]:
    questions: List[str] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            question = record.get("qa", {}).get("question")
            if isinstance(question, str) and question.strip():
                questions.append(question.strip())
    return questions


def sample_questions(questions: List[str], k: int, rng: random.Random, source_name: str) -> List[str]:
    if len(questions) < k:
        raise ValueError(
            f"{source_name} has only {len(questions)} questions, but {k} samples were requested."
        )
    return rng.sample(questions, k)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    remove_questions = load_questions(args.remove_path)
    retain_questions = load_questions(args.retain_path)

    sampled_remove = sample_questions(
        remove_questions, args.samples_per_label, rng, "remove dataset"
    )
    sampled_retain = sample_questions(
        retain_questions, args.samples_per_label, rng, "retain dataset"
    )

    output_data = {
        "bio_forget": sampled_remove,
        "neutral": sampled_retain,
    }

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

    print(
        f"Wrote {2 * args.samples_per_label} rows to {args.output_path} "
        f"({args.samples_per_label} per label)."
    )


if __name__ == "__main__":
    main()
