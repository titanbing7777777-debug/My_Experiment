import json
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple


Quadruple = Tuple[str, str, str, str]


_REPAIR_KEY_PATTERN = re.compile(r'([{,]\s*)([A-Za-z0-9_]+)(\s*:)')


def _repair_json_like(text: str) -> str:
    """Quote bare object keys so the string becomes valid JSON."""

    return _REPAIR_KEY_PATTERN.sub(lambda m: f'{m.group(1)}"{m.group(2)}"{m.group(3)}', text)


def _safe_load_object(raw: str) -> Tuple[Dict[str, Iterable[Dict[str, str]]], bool]:
    if not raw:
        return {"quadruples": []}, False

    try:
        return json.loads(raw), True
    except json.JSONDecodeError:
        repaired = _repair_json_like(raw)
        try:
            return json.loads(repaired), False
        except json.JSONDecodeError:
            return {"quadruples": []}, False


def _normalise_quadruples(obj: Dict[str, Iterable[Dict[str, str]]]) -> Set[Quadruple]:
    quadruples: Set[Quadruple] = set()
    for item in obj.get("quadruples", []) or []:
        values: List[str] = []
        for key in ("target", "aspect", "opinion", "sentiment"):
            value = item.get(key, "") if isinstance(item, dict) else ""
            if value is None:
                value = ""
            values.append(str(value).strip())
        quadruples.add(tuple(values))
    return quadruples


def evaluate(predictions: Sequence[str], references: Sequence[str]) -> Tuple[int, int, int, List[int]]:
    gold_total = 0
    pred_total = 0
    correct_total = 0
    failed_lines: List[int] = []

    for idx, (pred_raw, ref_raw) in enumerate(zip(predictions, references), start=1):
        pred_obj, pred_ok = _safe_load_object(pred_raw)
        ref_obj, ref_ok = _safe_load_object(ref_raw)

        pred_set = _normalise_quadruples(pred_obj)
        ref_set = _normalise_quadruples(ref_obj)

        if not (pred_ok and ref_ok):
            failed_lines.append(idx)

        pred_total += len(pred_set)
        gold_total += len(ref_set)
        correct_total += len(pred_set & ref_set)

    return correct_total, pred_total, gold_total, failed_lines


def analysis1(result_path: Path) -> None:
    if not result_path.is_file():
        raise FileNotFoundError(f"Result file not found: {result_path}")

    responses: List[str] = []
    labels: List[str] = []
    malformed_records: List[int] = []

    with result_path.open("r", encoding="utf-8") as source:
        for line_number, raw_line in enumerate(source, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue

            try:
                record = json.loads(stripped)
            except json.JSONDecodeError:
                malformed_records.append(line_number)
                continue

            responses.append(record.get("response", ""))
            labels.append(record.get("labels", ""))

    correct, pred_total, gold_total, failed_lines = evaluate(responses, labels)

    precision = correct / pred_total if pred_total else 0.0
    recall = correct / gold_total if gold_total else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision and recall else 0.0

    print(f"Samples processed: {len(responses)}")
    print(f"Gold quadruples: {gold_total}")
    print(f"Predicted quadruples: {pred_total}")
    print(f"Correct predictions: {correct}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")
    if failed_lines:
        print(f"Lines with malformed quadruple payloads: {failed_lines}")
    if malformed_records:
        print(f"Malformed JSON records skipped: {malformed_records}")

def analysis2(result_path: Path) -> None:
    if not result_path.is_file():
        raise FileNotFoundError(f"Result file not found: {result_path}")

    def safe_load_list(raw: str) -> Tuple[List[Dict[str, str]], bool]:
        if raw is None:
            return [], False
        if isinstance(raw, list):
            return raw, True
        if not raw:
            return [], False
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                return list(obj.get("quadruples", []) or []), True
            if isinstance(obj, list):
                return obj, True
            return [], False
        except json.JSONDecodeError:
            repaired = _repair_json_like(str(raw))
            try:
                obj = json.loads(repaired)
                if isinstance(obj, dict):
                    return list(obj.get("quadruples", []) or []), False
                if isinstance(obj, list):
                    return obj, False
            except json.JSONDecodeError:
                return [], False
        return [], False

    def normalise_list(items: List[Dict[str, str]]) -> Set[Quadruple]:
        quadruples: Set[Quadruple] = set()
        for item in items:
            if not isinstance(item, dict):
                continue
            values: List[str] = []
            for key in ("target", "aspect", "opinion", "sentiment"):
                value = item.get(key, "")
                if value is None:
                    value = ""
                values.append(str(value).strip())
            quadruples.add(tuple(values))
        return quadruples

    responses: List[str] = []
    labels: List[str] = []
    malformed_records: List[int] = []

    with result_path.open("r", encoding="utf-8") as source:
        for line_number, raw_line in enumerate(source, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue

            try:
                record = json.loads(stripped)
            except json.JSONDecodeError:
                malformed_records.append(line_number)
                continue

            responses.append(record.get("response", ""))
            labels.append(record.get("labels", ""))

    gold_total = 0
    pred_total = 0
    correct_total = 0
    failed_lines: List[int] = []

    for idx, (pred_raw, ref_raw) in enumerate(zip(responses, labels), start=1):
        pred_list, pred_ok = safe_load_list(pred_raw)
        ref_list, ref_ok = safe_load_list(ref_raw)

        pred_set = normalise_list(pred_list)
        ref_set = normalise_list(ref_list)

        if not (pred_ok and ref_ok):
            failed_lines.append(idx)

        pred_total += len(pred_set)
        gold_total += len(ref_set)
        correct_total += len(pred_set & ref_set)

    precision = correct_total / pred_total if pred_total else 0.0
    recall = correct_total / gold_total if gold_total else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision and recall else 0.0

    print(f"Samples processed: {len(responses)}")
    print(f"Gold quadruples: {gold_total}")
    print(f"Predicted quadruples: {pred_total}")
    print(f"Correct predictions: {correct_total}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")
    if failed_lines:
        print(f"Lines with malformed quadruple payloads: {failed_lines}")
    if malformed_records:
        print(f"Malformed JSON records skipped: {malformed_records}")

def titan_analysis(result_path: Path) -> None:
    if not result_path.is_file():
        raise FileNotFoundError(f"Result file not found: {result_path}")

    total_quads = 0
    valid_lines = 0
    pred_set = set()
    gold_set = set()

    with result_path.open('r', encoding='utf-8') as f:
        for line_id, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                response = data.get("response", "")
                label = data.get("labels", "")
                pred_labels = json.loads(response)
                gold_labels = json.loads(label)
                total_quads += len(gold_labels)
                for pred_label in pred_labels:
                    values = (
                        str(pred_label.get("target", "")).strip(),
                        str(pred_label.get("aspect", "")).strip(),
                        str(pred_label.get("opinion", "")).strip(),
                        str(pred_label.get("sentiment", "")).strip(),
                    )
                    pred_set.add(values)
                for gold_label in gold_labels:
                    values = (
                        str(gold_label.get("target", "")).strip(),
                        str(gold_label.get("aspect", "")).strip(),
                        str(gold_label.get("opinion", "")).strip(),
                        str(gold_label.get("sentiment", "")).strip(),
                    )
                    gold_set.add(values)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line at line {line_id}")

    correct_set = pred_set & gold_set
    print(f"Total quadruples in gold data: {total_quads}")
    print(f"Predicted quadruples: {len(pred_set)}")
    print(f"Gold quadruples: {len(gold_set)}")
    print(f"Correct quadruples: {len(correct_set)}")

                


if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    result_path = current_dir / "results" / "Qwen2-7B-Instruct" / "v0-20260205-105228" / "checkpoint-72" / "test_result.jsonl"
    titan_analysis(result_path)