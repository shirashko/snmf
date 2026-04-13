import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a labeled bio dataset JSON by sampling examples for "
            "bio_forget / bio_retain / neutral labels."
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
        help="Path to retain split JSONL (mapped to label 'bio_retain').",
    )
    parser.add_argument(
        "--neutral-path",
        type=Path,
        default=Path("data/valid_eng.jsonl"),
        help="Path to neutral JSONL (mapped to label 'neutral').",
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
        default=400,
        help="How many samples to draw for each label.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for reproducible sampling.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Keep only the first N whitespace tokens from each sampled text (0 disables truncation).",
    )
    return parser.parse_args()


def load_texts(jsonl_path: Path) -> List[str]:
    texts: List[str] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            text = (
                record.get("qa", {}).get("question")
                if isinstance(record.get("qa"), dict)
                else None
            )
            if not isinstance(text, str) or not text.strip():
                text = record.get("question")
            if not isinstance(text, str) or not text.strip():
                text = record.get("text")
            if isinstance(text, str) and text.strip():
                texts.append(text.strip())
    return texts


def load_qa_questions_and_answers(jsonl_path: Path) -> Tuple[List[str], List[str]]:
    questions: List[str] = []
    answers: List[str] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            qa = record.get("qa", {})
            if not isinstance(qa, dict):
                continue

            question = qa.get("question")
            answer = qa.get("answer")
            if isinstance(question, str) and question.strip():
                questions.append(question.strip())
            if isinstance(answer, str) and answer.strip():
                answers.append(answer.strip())
    return questions, answers


def sample_texts(texts: List[str], k: int, rng: random.Random, source_name: str) -> List[str]:
    if len(texts) < k:
        raise ValueError(
            f"{source_name} has only {len(texts)} rows, but {k} samples were requested."
        )
    return rng.sample(texts, k)


def sample_half_questions_half_answers(
    questions: List[str], answers: List[str], k: int, rng: random.Random, source_name: str
) -> List[str]:
    question_k = k // 2
    answer_k = k - question_k
    if len(questions) < question_k:
        raise ValueError(
            f"{source_name} has only {len(questions)} questions, but {question_k} are required."
        )
    if len(answers) < answer_k:
        raise ValueError(
            f"{source_name} has only {len(answers)} answers, but {answer_k} are required."
        )
    sampled = rng.sample(questions, question_k) + rng.sample(answers, answer_k)
    rng.shuffle(sampled)
    return sampled


def truncate_to_first_tokens(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return text
    tokens = text.split()
    return " ".join(tokens[:max_tokens])


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    label_to_source: Dict[str, Path] = {
        "bio_forget": args.remove_path,
        "bio_retain": args.retain_path,
        "neutral": args.neutral_path,
    }

    output_data: Dict[str, List[str]] = {}
    for label, source_path in label_to_source.items():
        if label in {"bio_forget", "bio_retain"}:
            questions, answers = load_qa_questions_and_answers(source_path)
            sampled_texts = sample_half_questions_half_answers(
                questions,
                answers,
                args.samples_per_label,
                rng,
                f"{label} dataset",
            )
        else:
            source_texts = load_texts(source_path)
            sampled_texts = sample_texts(
                source_texts, args.samples_per_label, rng, f"{label} dataset"
            )
        sampled_texts = [
            truncate_to_first_tokens(text, args.max_tokens) for text in sampled_texts
        ]
        output_data[label] = sampled_texts

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

    print(
        f"Wrote {3 * args.samples_per_label} rows to {args.output_path} "
        f"({args.samples_per_label} per label)."
    )


if __name__ == "__main__":
    main()
