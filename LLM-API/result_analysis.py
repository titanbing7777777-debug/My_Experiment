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

def read_result(result_path):
    with open(result_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for line_number, line in enumerate(lines):
        try:
            results = [json.loads(line.strip())]
            print(type(results))
            response = json.loads(results[0]["response"])["quadruples"]
            label = json.loads(results[0]["gold"])["quadruples"]
            print(f"Line {line_number}:")
            print(f"Predicted Quadruples:{response}")
            print(f"Gold Quadruples:{label}")
            cmd = input("Press Enter to continue (q to quit): ")
            if cmd.lower() == 'q':
                break
        except json.JSONDecodeError:
            print(f"Malformed JSON at line {line_number}")

def error_type_analysis(result_path):
    with open(result_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    analysis = {
        "TP": 0,
        "FN_Missing": 0,
        "FP_Hallucination": 0,
        "Sentiment_Error": 0,
        "Target_Error": 0,
        "Aspect_Error": 0,
        "Opinion_Error": 0
    }

    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue

        raw_gold = data.get("gold", "")
        raw_pred = data.get("response", "")

        # 解析 Gold
        if raw_gold == "statement-non-opinion":
            gold_quads = []
        else:
            try:
                gold_json = json.loads(raw_gold)
                if isinstance(gold_json, dict):
                    gold_quads = gold_json.get("quadruples", [])
                else:
                    gold_quads = []
            except json.JSONDecodeError:
                gold_quads = []
        
        # 解析 Prediction (使用 process_response)
        pred_quads = process_response(raw_pred)

        # 转换为 tuple 集合方便比较 (target, aspect, opinion, sentiment)
        gold_set = set()
        for q in gold_quads:
            if isinstance(q, dict):
                gold_set.add((q.get("target"), q.get("aspect"), q.get("opinion"), q.get("sentiment")))
        
        pred_set = set()
        for q in pred_quads:
            if isinstance(q, dict):
                pred_set.add((q.get("target"), q.get("aspect"), q.get("opinion"), q.get("sentiment")))

        # 统计
        tp_set = gold_set.intersection(pred_set)
        analysis["TP"] += len(tp_set)

        missed = list(gold_set - pred_set)
        spurious = list(pred_set - gold_set)

        matched_spurious_indices = set()

        for g in missed:
            # g: (target, aspect, opinion, sentiment)
            # 尝试在 spurious 中寻找"修改"过的版本
            matched_type = None
            matched_idx = -1
            
            for idx, p in enumerate(spurious):
                if idx in matched_spurious_indices:
                    continue
                
                # 检查不同类型的错误 (优先级: Sentiment > Opinion > Aspect > Target)
                # 1. Sentiment Error: T, A, O 相同
                if g[0] == p[0] and g[1] == p[1] and g[2] == p[2]:
                    matched_type = "Sentiment_Error"
                    matched_idx = idx
                    break # 最强匹配，直接跳出
                
                # 2. Opinion Error: T, A, S 相同
                elif g[0] == p[0] and g[1] == p[1] and g[3] == p[3]:
                    if matched_type is None:
                        matched_type = "Opinion_Error"
                        matched_idx = idx
                
                # 3. Aspect Error: T, O, S 相同
                elif g[0] == p[0] and g[2] == p[2] and g[3] == p[3]:
                    if matched_type is None:
                        matched_type = "Aspect_Error"
                        matched_idx = idx

                # 4. Target Error: A, O, S 相同
                elif g[1] == p[1] and g[2] == p[2] and g[3] == p[3]:
                    if matched_type is None:
                        matched_type = "Target_Error"
                        matched_idx = idx
            
            if matched_type:
                analysis[matched_type] += 1
                matched_spurious_indices.add(matched_idx)
            else:
                analysis["FN_Missing"] += 1
        
        # 剩下的 spurious 就是纯粹的幻觉
        analysis["FP_Hallucination"] += (len(spurious) - len(matched_spurious_indices))

    print("Error Analysis Result:")
    print(json.dumps(analysis, indent=4, ensure_ascii=False))

if __name__ == "__main__":
    BASE = Path(__file__).resolve().parent.parent
    data_path = BASE / "LLM-API" / "result" / "en" / "result(five-shot)(deepseek-ai_DeepSeek-V3.2).json"
    # read_result(data_path)
    error_type_analysis(data_path)