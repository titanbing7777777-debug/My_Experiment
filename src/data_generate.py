import json
from pathlib import Path
import os

def data_utterance():
    """
        Input: Utterance
        Output: Quadruples
    """

def process_data1(data_path,outdir_path,mode):
    """
        Input: A set of utterances with reply relations
        Output: Quadruples(in current utterance)(last term in this utterance)
    """
    outdir_path = Path(outdir_path)
    outdir_path.mkdir(parents=True, exist_ok=True)
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for data_id, d in enumerate(data):
        speakers = d["speakers"]
        replies = d["replies"]
        utterances = d["sentences"]
        dialogue_length = len(utterances)

        # 判断是否为最后一句
        final_utterance = [True for _ in range(dialogue_length)]
        for r in replies:
            if r != -1:
                final_utterance[r] = False
        # 获得这个对话样本中的回复链
        reply_chains = []
        for f_id, f in enumerate(final_utterance):
            if f:
                chain = []
                current_utterance_id = f_id
                while(current_utterance_id != -1):
                    chain.append(current_utterance_id)
                    current_utterance_id = replies[current_utterance_id]
                chain.reverse()
                reply_chains.append(chain)

        # 找到每句话的边界
        word_spilid = []
        for i in range(dialogue_length):
            if i == 0:
                word_spilid.append(len(utterances[i].split()))
            else:
                word_spilid.append(word_spilid[i-1] + len(utterances[i].split()))
        # 筛选四元组
        triples = d["triplets"]
        quadruple = []
        quadruple_u = [[] for _ in range(dialogue_length)]
        quadruple.extend(
            (t[7], t[8], t[9], t[6], t[0], t[2], t[4])
            for t in triples
            if t[6] != -1 and all(x is not None and x != '' for x in (t[7], t[8], t[9]))
        )
        # 对四元组进行分类
        res = {'pos': 'pos', 'neg': 'neg'}
        for q in quadruple:
            target_sentence_index = -1
            aspect_sentence_index = -1
            opinion_sentence_index = -1
            for uid, ws in enumerate(word_spilid):
                if q[4] < ws and target_sentence_index == -1:
                    target_sentence_index = uid
                if q[5] < ws and aspect_sentence_index == -1:
                    aspect_sentence_index = uid
                if q[6] < ws and opinion_sentence_index == -1:
                    opinion_sentence_index = uid
            if target_sentence_index != -1 and aspect_sentence_index != -1 and opinion_sentence_index != -1:
                max_index = max(target_sentence_index, aspect_sentence_index, opinion_sentence_index)
                current_quadruple = {
                    "target":q[0],
                    "target_sentence_id":target_sentence_index,
                    "aspect":q[1],
                    "aspect_sentence_id":aspect_sentence_index,
                    "opinion":q[2],
                    "opinion_sentence_id":opinion_sentence_index,
                    "sentiment":res.get(q[3], 'other')
                }
                quadruple_u[max_index].append(current_quadruple)
            else:
                raise ValueError("Quadruple sentence index error!")

        for chain_id, chain in enumerate(reply_chains):
            data = []
            for u_id in chain:
                data.append({
                    "utterance":f"<u{u_id}>{utterances[u_id]}(reply_to:u{replies[u_id]})",
                    "quadruples":quadruple_u[u_id]
                })
                dict_item = {
                    "sample_id":f"{data_id}_{chain_id}_{u_id}",
                    "data":data
                }
                output_file = outdir_path / f"{mode}.jsonl"
                with open(output_file, "a", encoding="utf-8") as f_out:
                    f_out.write(json.dumps(dict_item, ensure_ascii=False) + "\n")

def process_data2(data_path, outdir_path, mode):
    """
        Query1: Utterance with target term
        Response1: {"target":target term, "aspect":None, "opinion":None, "sentiment":None}
        Query2: Utterance with aspect term
        Response2: {"target":target term, "aspect":aspect term, "opinion":None, "sentiment":None}
        Query3: Utterance with opinion term
        Response3: {"target":target term, "aspect":aspect term, "opinion":opinion term, "sentiment":pos/neg/other}
    """
    outdir_path = Path(outdir_path)
    outdir_path.mkdir(parents=True, exist_ok=True)
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for data_id, d in enumerate(data):
        speakers = d["speakers"]
        replies = d["replies"]
        utterances = d["sentences"]
        dialogue_length = len(utterances)

        # 判断是否为最后一句
        final_utterance = [True for _ in range(dialogue_length)]
        for r in replies:
            if r != -1:
                final_utterance[r] = False
        # 获得这个对话样本中的回复链
        reply_chains = []
        for f_id, f in enumerate(final_utterance):
            if f:
                chain = []
                current_utterance_id = f_id
                while(current_utterance_id != -1):
                    chain.append(current_utterance_id)
                    current_utterance_id = replies[current_utterance_id]
                chain.reverse()
                reply_chains.append(chain)

        # 找到每句话的边界
        word_spilid = []
        for i in range(dialogue_length):
            if i == 0:
                word_spilid.append(len(utterances[i].split()))
            else:
                word_spilid.append(word_spilid[i-1] + len(utterances[i].split()))
        # 筛选四元组
        triples = d["triplets"]
        quadruple = []
        quadruple_u = [[] for _ in range(dialogue_length)]
        quadruple.extend(
            (t[7], t[8], t[9], t[6], t[0], t[2], t[4])
            for t in triples
            if t[6] != -1 and all(x is not None and x != '' for x in (t[7], t[8], t[9]))
        )

        # 对四元组进行分类
        res = {'pos': 'pos', 'neg': 'neg'}
        for q in quadruple:
            target_sentence_index = -1
            aspect_sentence_index = -1
            opinion_sentence_index = -1
            # 找到各个元素所在句子的索引
            for uid, ws in enumerate(word_spilid):
                if q[4] < ws and target_sentence_index == -1:
                    target_sentence_index = uid
                if q[5] < ws and aspect_sentence_index == -1:
                    aspect_sentence_index = uid
                if q[6] < ws and opinion_sentence_index == -1:
                    opinion_sentence_index = uid
        
            if target_sentence_index == -1 or aspect_sentence_index == -1 or opinion_sentence_index == -1:
                raise ValueError("Quadruple sentence index error!")

            init_quad = {
                "target":"",
                "target_sentence_id":-1,
                "aspect":"",
                "aspect_sentence_id":-1,
                "opinion":"",
                "opinion_sentence_id":-1,
                "sentiment":""
            }

            for i in range(dialogue_length):
                if i in (target_sentence_index, aspect_sentence_index, opinion_sentence_index):
                    if i == target_sentence_index:
                        init_quad["target"] = q[0]
                        init_quad["target_sentence_id"] = target_sentence_index
                    if i == aspect_sentence_index:
                        init_quad["aspect"] = q[1]
                        init_quad["aspect_sentence_id"] = aspect_sentence_index
                    if i == opinion_sentence_index:
                        init_quad["opinion"] = q[2]
                        init_quad["opinion_sentence_id"] = opinion_sentence_index
                    if init_quad["target"] != "" and init_quad["aspect"] != "" and init_quad["opinion"] != "":
                        init_quad["sentiment"] = res.get(q[3], 'other')
                    quadruple_u[i].append(init_quad.copy())

        for chain_id, chain in enumerate(reply_chains):
            data = []
            for u_id in chain:
                data.append({
                    "utterance":f"<u{u_id}>{utterances[u_id]}(reply_to:u{replies[u_id]})",
                    "quadruples":quadruple_u[u_id]
                })
                dict_item = {
                    "sample_id":f"{data_id}_{chain_id}_{u_id}",
                    "data":data
                }
                output_file = outdir_path / f"{mode}.jsonl"
                with open(output_file, "a", encoding="utf-8") as f_out:
                    f_out.write(json.dumps(dict_item, ensure_ascii=False) + "\n")

def exp(origin_path):
    quad_set = set()
    with open(origin_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    total_quads = 0
    for line in lines:
        json_line = json.loads(line.strip())
        data = json_line["data"]
        for d in data:
            quads = d["quadruples"]
            total_quads += len(quads)
            for q in quads:
                quad_set.add((q["target"], q["aspect"], q["opinion"], q["sentiment"]))
    print(f"Total quadruples: {total_quads}")
    print(f"Unique quadruples: {len(quad_set)}")

if __name__ == "__main__":

    current_dir = os.path.dirname(os.path.abspath(__file__))
    mode = "valid"

    print("current directory:", current_dir) # /workspace/My_Experiment/src

    origin_path = os.path.join(current_dir, "..", "dataset", "jsons_en", f"{mode}.json")

    target_path = os.path.join(current_dir, "..", "data3.0", "en")

    process_data2(origin_path,target_path,mode)

    # exp(target_path + f"/{mode}.jsonl")
    