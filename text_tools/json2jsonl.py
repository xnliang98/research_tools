import json
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-file", type=str, default="/Users/bytedance/Downloads/ChatQilin-zh-BZ1K/train.50k.src_tgt.text.json")
    parser.add_argument("--out-file", type=str, default="/Users/bytedance/Downloads/ChatQilin-zh-BZ1K/train.50k.src_tgt.text.jsonl")
    args = parser.parse_args()

    
    print(args)
    args = vars(args)
    data = json.load(open(args['in_file']))
    with open(args['out_file'], "w") as fw:
        for d in data:
            fw.writelines(f"{json.dumps(d, ensure_ascii=False)}\n")
    print("Done!")

