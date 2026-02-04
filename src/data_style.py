import json
import pandas as pd
from pathlib import Path
import os


def json2csv(json_filepath, csv_filepath):

    # Load JSON data
    data = []
    count = 0
    with open(json_filepath, 'r', encoding='utf-8') as json_file:
        for line in json_file:
            if line.strip():
                print(line)
                json_line = json.loads(line)
                if json_line["target"] == "statement-non-opinion":
                    json_line["target"] = "notarget:none:none:none"
                else:
                    target_data = json.loads(json_line["target"])
                    quadruples = target_data.get("quadruples", [])
                    count += len(quadruples)
                    # 注意：根据 JSON 内容，字段名是 target, aspect, opinion, sentiment
                    # 并没有 category，如果不确定可以用 get
                    json_line["target"] = ", ".join([
                        f"{item.get('target', 'none')}:{item.get('aspect', 'none')}:{item.get('opinion', 'none')}:{item.get('sentiment', 'none')}" 
                        for item in quadruples
                    ])
                data.append(json_line)

    print(f"Total target count: {count}")

    if data:
        print(data[0])

    # Normalize JSON data to flat table
    df = pd.json_normalize(data)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(csv_filepath)), exist_ok=True)

    # Save to CSV
    df.to_csv(csv_filepath, index=False, encoding='utf-8')

def process_data1(origin_path, target_path, mode):
    """
    {"messages": [
        {"role": "system", "content": "<system>"}, 
        {"role": "user", "content": "<query1>"}, 
        {"role": "assistant", "content": "<response1>"},
    """
    with open(origin_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    output_file = Path(target_path) / f"{mode}.jsonl"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f_out:
        for line in lines:
            line = line.strip()
            if not line: continue
            json_line = json.loads(line)
            dict_item = {
                "messages":[
                    {"role":"system","content":"You are a helpful assistant that extracts quadruples from a utterance."},
                    {"role":"user","content":json_line["input"]},
                    {"role":"assistant","content":json_line["target"] if json_line["target"]!="statement-non-opinion" else "{quadruples: []}"}
                ]
            }
            f_out.write(json.dumps(dict_item) + "\n")

def process_data2(origin_path, target_path, mode):
    """
    {"messages": [
        {"role": "system", "content": "<system>"}, 
        {"role": "user", "content": "<utterance1>"}, 
        {"role": "assistant", "content": "<response1>"},
        {"role": "user", "content": "<utterance2>"}, 
        {"role": "assistant", "content": "<response2>"},
        ...
    """
    with open(origin_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    output_file = Path(target_path) / f"{mode}.jsonl"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f_out:
        for line in lines:
            line = line.strip()
            if not line: continue
            json_line = json.loads(line)
            chain_length = json_line["chain_length"]
            data = json_line["data"]
            dict_item = {
                "messages":[
                    {"role":"system","content":"You are a helpful assistant that extracts quadruples from a series of utterances in a dialogue."}
                ]
            }
            for i in range(chain_length):
                dict_item["messages"].append({"role":"user","content":data[i]["utterance"]})
                dict_item["messages"].append({"role":"assistant","content":json.dumps(data[i]["quadruples"], ensure_ascii=False)})
            
            f_out.write(json.dumps(dict_item) + "\n")




if __name__ == "__main__":

    current_dir = os.path.dirname(os.path.abspath(__file__))
    mode = "valid"

    print("current directory:", current_dir) # /workspace/My_Experiment/src

    origin_path = os.path.join(current_dir, "..", "data(reply_chain)(all)", "en", f"{mode}.jsonl")

    target_path = os.path.join(current_dir, "..", "Fine_tune", "Dataset(reply_chain)", "DiaASQ")

    process_data2(origin_path, target_path, mode)