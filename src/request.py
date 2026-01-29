
from openai import OpenAI
import json
import numpy as np
from pathlib import Path
from utils import task_definer

client = OpenAI(api_key="sk-bhihrehevhxpelpzuvpvyplrpnomnsknhvtkqvaefigpdnjq", 
                base_url="https://api.siliconflow.cn/v1")

def sent_request(data_path, outdir_path, model, shot_count=0):
    examples = []
    
    if shot_count > 0:
        # 加载训练集作为 few-shot 示例
        BASE = Path(__file__).resolve().parent.parent
        train_path = BASE / "data(zero_shot)" / "en" / "train.json"
        
        with open(train_path, "r", encoding="utf-8") as f:
            train_lines = f.readlines()
        
        train_count = len(train_lines)
        # 从训练集中随机选择 shot_count 个样本
        numbers = np.random.choice(train_count, size=shot_count, replace=False)

        for e_id, n in enumerate(numbers):
            json_line = json.loads(train_lines[n])
            examples.append(f"###Example{e_id+1}:\n###Input: {json_line['input']}\n###Output: {json_line['target']}\n")
        
        print(f"{shot_count}-shot examples loaded")

    # 确保输出目录存在
    outdir_path = Path(outdir_path)
    outdir_path.mkdir(parents=True, exist_ok=True)
    
    # 替换模型名中的斜杠，防止路径解析错误
    safe_model_name = model.replace("/", "_")
    shot_type = f"{shot_count}-shot" if shot_count > 0 else "zero-shot"
    output_filename = f"result({shot_type})({safe_model_name}).json"
    
    with open(data_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    print(f"Processing {len(lines)} test samples with {shot_type}...")
    
    # 使用 'w' 模式重写文件，避免重复数据
    processed_count = 0
    with open(outdir_path / output_filename, "w", encoding="utf-8") as f_out:
        for line in lines:
            line = line.strip()
            if not line:
                continue
            json_line = json.loads(line)
            input_text = "".join(e for e in examples) + f"###Input:\n{json_line['input']}\n###Output:"
            
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {'role': 'system', 
                    'content': task_definer("quadruples")},
                    {'role': 'user', 
                    'content': input_text}
                ],
                stream=False
            )
            print(f"ID: {json_line['sample_id']}")
            print("Input:", json_line['input'])
            print("Response:", response.choices[0].message.content)
            
            result_dict = {
                "sample_id": json_line['sample_id'],
                "input": input_text,
                "gold": json_line['target'],
                "response": response.choices[0].message.content
            }

            f_out.write(json.dumps(result_dict, ensure_ascii=False) + "\n")
            f_out.flush()  # 确保实时写入磁盘


if __name__ == "__main__":
    BASE = Path(__file__).resolve().parent.parent
    data_path = BASE / "data(zero_shot)" / "en" / "test.json"
    outdir_path = BASE / "result" / "en"
    model = "Pro/deepseek-ai/DeepSeek-R1"
    
    # 设置示例数量：0 表示 zero-shot，>0 表示 few-shot
    # 例如：shot_count=0 (zero-shot), shot_count=5 (five-shot), shot_count=10 (ten-shot)
    shot_count = 5
    sent_request(data_path, outdir_path, model, shot_count)