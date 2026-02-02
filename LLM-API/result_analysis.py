import json
import re
from pathlib import Path

def process_response(response_text):
    """
    解析模型返回的 response 文本，提取四元组列表
    处理多种可能的格式：
    1. 纯 JSON: {"quadruples": [...]}
    2. Markdown 代码块: ```json\n{...}\n```
    3. statement-non-opinion
    4. 空数组: {"quadruples": []}
    """
    response_text = response_text.strip()
    
    # 处理 statement-non-opinion 情况
    if response_text == "statement-non-opinion":
        return []
    
    # 移除可能的 markdown 代码块标记
    if response_text.startswith("```"):
        # 移除开头的 ```json 或 ```
        response_text = re.sub(r'^```(?:json)?\s*\n?', '', response_text)
        # 移除结尾的 ```
        response_text = re.sub(r'\n?```\s*$', '', response_text)
        response_text = response_text.strip()
    
    # 尝试解析 JSON
    try:
        data = json.loads(response_text)
        
        # 提取 quadruples 字段
        if isinstance(data, dict):
            quadruples = data.get("quadruples", [])
            if isinstance(quadruples, list):
                # 过滤掉包含 null 或无效字段的四元组
                valid_quadruples = []
                for q in quadruples:
                    if isinstance(q, dict):
                        # 检查必需字段是否存在且非 None
                        if (q.get("target") is not None and 
                            q.get("aspect") is not None and 
                            q.get("opinion") is not None and 
                            q.get("sentiment") is not None):
                            valid_quadruples.append(q)
                return valid_quadruples
        return []
    except json.JSONDecodeError as e:
        # JSON 解析失败，返回空列表
        print(f"Warning: Failed to parse response: {e}")
        print(f"Response text: {response_text[:100]}...")
        return []

def result_analysis1(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    gold_count = 0
    pred_count = 0
    correct_count = 0

    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        json_line = json.loads(line)

        gold = json_line["gold"]
        response = json_line["response"]
        
        # 处理 gold 数据
        if gold == "statement-non-opinion":
            gold_data = []
        else:
            gold_data = json.loads(gold)["quadruples"]
        
        pred_data = process_response(response)
        
        gold_set = set()
        pred_set = set()
        for item in gold_data:
            gold_set.add((item["target"], item["aspect"], item["opinion"], item["sentiment"]))
        for item in pred_data:
            pred_set.add((item["target"], item["aspect"], item["opinion"], item["sentiment"]))
        
        correct_set = gold_set.intersection(pred_set)
        gold_count += len(gold_set)
        pred_count += len(pred_set)
        correct_count += len(correct_set)
        

    
    precision = correct_count / pred_count if pred_count > 0 else 0.0
    recall = correct_count / gold_count if gold_count > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    print(f"Precision: {precision:.4f} ({correct_count}/{pred_count})")
    print(f"Recall: {recall:.4f} ({correct_count}/{gold_count})")
    print(f"F1 Score: {f1_score:.4f}")

def _normalize_sentiment(value: str):
    if value is None:
        return None
    v = re.sub(r'[^a-zA-Z]+', '', str(value)).lower()
    if v in {"pos", "neg", "other"}:
        return v
    if v in {"positive", "posit", "positve"}:
        return "pos"
    if v in {"negative", "negat", "negativ"}:
        return "neg"
    if v in {"neutral", "none"}:
        return "other"
    return None

def _parse_quadruples_text(text: str):
    """
    解析形如:
      target:aspect:opinion:sentiment, ...
    允许逗号/分号/竖线分隔
    """
    if text is None:
        return []

    t = text.strip()
    if not t or t == "statement-non-opinion" or t == "notarget:none:none:none":
        return []

    if "output:" in t:
        t = t.split("output:")[-1].strip()

    parts = re.split(r'\s*(?:,|;|\|)\s*', t)
    results = []

    for p in parts:
        p = p.strip()
        if not p or p == "notarget:none:none:none":
            continue

        fields = p.split(":", 3)
        if len(fields) < 4:
            continue

        target, aspect, opinion, sentiment = [x.strip() for x in fields]
        sent = _normalize_sentiment(sentiment)
        if not sent:
            continue
        if not target or target == "notarget":
            continue
        if not aspect or not opinion:
            continue

        results.append({
            "target": target,
            "aspect": aspect,
            "opinion": opinion,
            "sentiment": sent
        })

    return results

def result_analysis2(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    gold_count = 0
    pred_count = 0
    correct_count = 0

    for line in lines:
        line = line.strip()
        if not line:
            continue

        json_line = json.loads(line)
        gold = json_line.get("gold", "")
        response = json_line.get("response", "")

        gold_data = _parse_quadruples_text(gold)
        pred_data = _parse_quadruples_text(response)

        gold_set = set((i["target"], i["aspect"], i["opinion"], i["sentiment"]) for i in gold_data)
        pred_set = set((i["target"], i["aspect"], i["opinion"], i["sentiment"]) for i in pred_data)

        correct_set = gold_set.intersection(pred_set)
        gold_count += len(gold_set)
        pred_count += len(pred_set)
        correct_count += len(correct_set)

    precision = correct_count / pred_count if pred_count > 0 else 0.0
    recall = correct_count / gold_count if gold_count > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"Precision: {precision:.4f} ({correct_count}/{pred_count})")
    print(f"Recall: {recall:.4f} ({correct_count}/{gold_count})")
    print(f"F1 Score: {f1_score:.4f}")


if __name__ == "__main__":
    BASE = Path(__file__).resolve().parent.parent
    data_path = BASE / "LLM-API" / "result" / "en" / "result(InstructionSet2)(deepseek-ai_DeepSeek-V3.2).json"
    result_analysis2(data_path)