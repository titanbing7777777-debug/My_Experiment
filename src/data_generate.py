import json
from pathlib import Path
import os

def data_utterance():
    """
        Input: Utterance
        Output: Quadruples
    """

def data_reply_chain(data_path,outdir_path,mode):
    """
        Input: A set of utterances with reply relations
        Output: Quadruples(in current utterance)
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
                if q[4] < ws:
                    target_sentence_index = uid
                if q[5] < ws:
                    aspect_sentence_index = uid
                if q[6] < ws:
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
            dict_item = {
                "sample_id":f"{data_id}_{chain_id}",
                "chain_length":len(chain),
                "data":[]
            }
            for u_id in chain:
                dict_item["data"].append({
                    "utterance":f"<u{u_id}>{utterances[u_id]}(reply_to:u{replies[u_id]})",
                    "quadruples":quadruple_u[u_id]
                })

            output_file = outdir_path / f"{mode}.jsonl"
            with open(output_file, "a", encoding="utf-8") as f_out:
                f_out.write(json.dumps(dict_item, ensure_ascii=False) + "\n")


if __name__ == "__main__":

    current_dir = os.path.dirname(os.path.abspath(__file__))
    mode = "valid"

    print("current directory:", current_dir) # /workspace/My_Experiment/src

    origin_path = os.path.join(current_dir, "..", "dataset", "jsons_en", f"{mode}.json")

    target_path = os.path.join(current_dir, "..", "data(reply_chain)(all)", "en")

    data_reply_chain(origin_path,target_path,mode)
    