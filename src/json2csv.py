import json
import pandas as pd
import os


def json2csv(json_filepath, csv_filepath):

    # Load JSON data
    data = []
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
                    # 注意：根据 JSON 内容，字段名是 target, aspect, opinion, sentiment
                    # 并没有 category，如果不确定可以用 get
                    json_line["target"] = ", ".join([
                        f"{item.get('target', 'none')}:{item.get('aspect', 'none')}:{item.get('opinion', 'none')}:{item.get('sentiment', 'none')}" 
                        for item in quadruples
                    ])
                data.append(json_line)
    
    if data:
        print(data[0])

    # Normalize JSON data to flat table
    df = pd.json_normalize(data)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(csv_filepath)), exist_ok=True)

    # Save to CSV
    df.to_csv(csv_filepath, index=False, encoding='utf-8')

if __name__ == "__main__":

    current_dir = os.path.dirname(os.path.abspath(__file__))
    mode = "valid"

    json_path = os.path.join(current_dir, "..", "data(zero_shot)", "en", f"{mode}.json")

    csv_path = os.path.join(current_dir, "..", "InstructABSA-main", "Dataset", "DiaASQ", f"{mode}.csv")

    json2csv(json_path, csv_path)