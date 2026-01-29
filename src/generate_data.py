from pathlib import Path
import json
from utils import task_definer

def find_reply_chain(replies, current):
    chain = []
    while(current != -1):
        chain.append(current)
        current = replies[current]
    chain.reverse()
    return chain

def _normalize_target(value):
    """Ensure target field is always a string for JSONL loading."""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)

modes = ["train","valid","test"]
language = ["en","zh"]


BASE = Path(__file__).resolve().parent.parent  # /Fullmoon717/TaCoMoE
dataset_root = BASE / "dataset"

def process_data(dataset_path, outdir_path,mode,language):

    outdir_path = Path(outdir_path)
    outdir_path.mkdir(parents=True, exist_ok=True)
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    for data_id, d in enumerate(data):
        inputs_onebyone = []
        # print(d)
        speakers = d["speakers"]
        replies = d["replies"]
        sentences = d["sentences"]
        dialogue_length = len(d["sentences"]) 

        word_spilid = []

        for i in range(dialogue_length):
            if i == 0:
                word_spilid.append(len(sentences[i].split()))
            else:
                word_spilid.append(word_spilid[i-1] + len(sentences[i].split()))

        triples = d["triplets"]

        quadruple = []
        quadruple_u = [[] for _ in range(dialogue_length)]

        for t in triples:
            if t[7]!= "" and t[8] != "" and t[9] != "":
                quadruple.append((t[7],t[8],t[9],t[6],t[0],t[2],t[4]))

        res = {'pos': 'pos', 'neg': 'neg'}
        for q in quadruple:
            target_sentence_index = -1
            aspect_sentence_index = -1
            opinion_sentence_index = -1
            for uid, ws in enumerate(word_spilid):
                if target_sentence_index == -1 and q[4] < ws:
                    target_sentence_index = uid
                if aspect_sentence_index == -1 and q[5] < ws:
                    aspect_sentence_index = uid
                if opinion_sentence_index == -1 and q[6] < ws:
                    opinion_sentence_index = uid

            if (target_sentence_index == aspect_sentence_index == opinion_sentence_index) and target_sentence_index != -1:
                quadruple_u[target_sentence_index].append({
                    "target": q[0],
                    "aspect": q[1],
                    "opinion": q[2],
                    "sentiment": res.get(q[3], 'other')
                })

            
        for uid, Input in enumerate(sentences):
            if len(quadruple_u[uid]) == 0:
                target_value = "statement-non-opinion"
            else:
                target_value = {"quadruples": quadruple_u[uid]}
                # print(target_value)
                target_value = _normalize_target(target_value)
            data_dict = {
                "sample_id": f"{data_id}_{uid}",
                "input": Input,
                "target": target_value
            }

            with open(outdir_path / f"{mode}.json", "a", encoding="utf-8") as f:
                f.write(json.dumps(data_dict, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    m = "train"
    l = "en"
    process_data(dataset_root / f"jsons_{l}/{m}.json", BASE / f"data(five_shot)/{l}", m, l)