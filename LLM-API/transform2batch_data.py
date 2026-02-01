import json
from pathlib import Path
from utils import task_definer

def transform2batch_data(data_path, outdir_path, mode):
    with open(data_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    output_file = Path(outdir_path) / f"{mode}.jsonl"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f_out:
        for line in lines:
            line = line.strip()
            if not line: continue
            json_line = json.loads(line)
            dict_item = {
            "custom_id": json_line["sample_id"],
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "deepseek-ai/DeepSeek-V3.2",
                "messages": [
                    {
                        "role": "system",
                        "content": task_definer("quadruples")
                    },
                    {
                        "role": "user",
                        "content": f"###Input:\n{json_line['input']}\n###Output:"
                    }
                ],
                    "stream": False,
                    "max_tokens": 1583,
                    "thinking_budget": 32768
            }
        }

            f_out.write(json.dumps(dict_item, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    mode = "test"
    language = "en"
    BASE = Path(__file__).resolve().parent.parent
    data_path = BASE / "data" / language / f"{mode}.json"
    outdir_path = BASE / "Batch_test" / language
    transform2batch_data(data_path, outdir_path,mode)