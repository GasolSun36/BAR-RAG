import argparse
import json
import re
import string
from collections import Counter
from typing import List


def extract_answer_tag(text: str) -> str:
    """Extract content inside <answer>...</answer> from model output."""
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.DOTALL)
    if not m:
        return ""
    return m.group(1).strip()


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s: str) -> List[str]:
    if not s:
        return []
    return normalize_answer(s).split()


def compute_exact(a_gold: str, a_pred: str) -> float:
    return float(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold: str, a_pred: str) -> float:
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = Counter(gold_toks) & Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return float(gold_toks == pred_toks)
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def metric_max_over_ground_truths(
    metric_fn, prediction: str, ground_truths: List[str]
) -> float:
    if not ground_truths:
        ground_truths = [""]
    scores_for_ground_truths = []
    for gt in ground_truths:
        score = metric_fn(gt, prediction)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pred_file",
        type=str,
        required=True,
        help="JSONL file containing answers (list) and prediction",
    )
    parser.add_argument(
        "--print-error",
        action="store_true",
        help="Print all incorrect predictions",
    )
    args = parser.parse_args()

    total = 0
    exact_sum = 0.0
    f1_sum = 0.0
    missing_pred = 0
    no_answer_tag = 0

    with open(args.pred_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            total += 1

            answers = obj.get("answers", [])
            raw_prediction = obj.get("prediction", obj.get("pred", ""))

            if raw_prediction is None or raw_prediction == "":
                missing_pred += 1
                prediction = ""
            else:
                prediction = extract_answer_tag(raw_prediction)
                if not prediction:
                    no_answer_tag += 1
                    prediction = raw_prediction

            exact = metric_max_over_ground_truths(
                compute_exact, prediction, answers
            )
            f1 = metric_max_over_ground_truths(
                compute_f1, prediction, answers
            )

            if args.print_error and exact == 0:
                question = obj.get("question", "")
                print("ERROR SAMPLE:")
                print(f"Question: {question}")
                print(f"Ground Truth Answers: {answers}")
                print(f"Prediction: {prediction}")
                print("-" * 50)

            exact_sum += exact
            f1_sum += f1

    if total == 0:
        print("No examples found in pred_file.")
        return

    em = 100.0 * exact_sum / total
    f1 = 100.0 * f1_sum / total

    print(f"Total examples: {total}")
    print(f"Empty / missing predictions: {missing_pred}")
    print(f"No <answer> tag (fallback to raw): {no_answer_tag}")
    print(f"Exact Match (EM): {em:.2f}")
    print(f"F1: {f1:.2f}")


if __name__ == "__main__":
    main()
