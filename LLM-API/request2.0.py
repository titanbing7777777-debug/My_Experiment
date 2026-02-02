
from openai import OpenAI
import json
import numpy as np
import pandas as pd
from pathlib import Path
from utils import InstructionsHandler

client = OpenAI(api_key="sk-bhihrehevhxpelpzuvpvyplrpnomnsknhvtkqvaefigpdnjq", 
                base_url="https://api.siliconflow.cn/v1")

def sent_request(data_path, outdir_path, model, shot_count=0, set_mode = 1):

    handler = InstructionsHandler()
    
    if set_mode == 1:
        handler.load_instruction_set1(shot_count)
    elif set_mode == 2:
        handler.load_instruction_set2(shot_count)
    # 确保输出目录存在
    outdir_path = Path(outdir_path)
    outdir_path.mkdir(parents=True, exist_ok=True)
    
    # 替换模型名中的斜杠，防止路径解析错误
    safe_model_name = model.replace("/", "_")
    output_filename = f"result(InstructionSet{set_mode})({safe_model_name}).json"
    
    data_df = pd.read_csv(data_path)

    print(f"Processing {len(data_df)} test samples with {shot_count}_shot...")
    
    # 使用 'w' 模式重写文件，避免重复数据
    with open(outdir_path / output_filename, "w", encoding="utf-8") as f_out:
        for _, row in data_df.iterrows():
            json_line = row.to_dict()
            
            input_text = f"{handler.asqp['bos_instruct']}\ninput: {json_line['input']}{handler.asqp['eos_instruct']}"
            
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {'role': 'system', 
                    'content': 'You are a helpful assistant that extracts quadruples (target, aspect, opinion, sentiment) from the given text.'},
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
    data_path = BASE / "LLM-API" / "DiaASQ" / "test.csv"
    outdir_path = BASE / "LLM-API" / "result" / "en"
    model = "deepseek-ai/DeepSeek-V3.2"
    
    # 设置示例数量：0 表示 zero-shot，>0 表示 few-shot
    # 例如：shot_count=0 (zero-shot), shot_count=5 (five-shot), shot_count=10 (ten-shot)
    instruction_set = 1
    shot_count = 5
    sent_request(data_path, outdir_path, model, shot_count, instruction_set)