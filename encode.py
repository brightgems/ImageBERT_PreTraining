import argparse
import gzip
import json
import logging
import os
import sys
import torch
from tqdm import tqdm
from transformers import BertJapaneseTokenizer
from typing import Dict,List

sys.path.append(".")
import hashing

logging_fmt="%(asctime)s %(levelname)s: %(message)s"
logging.basicConfig(format=logging_fmt)
logger=logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

def load_contexts(context_filepath:str)->Dict[str,str]:
    """
    コンテキスト(Wikipedia記事)を読み込む。
    """
    contexts={}

    with gzip.open(context_filepath,mode="rt",encoding="utf-8") as r:
        for line in r:
            data = json.loads(line)

            title=data["title"]
            text=data["text"]

            contexts[title]=text

    return contexts

def encode_contexts(
    tokenizer:BertJapaneseTokenizer,
    contexts:Dict[str,str],
    max_seq_length:int,
    save_dir:str):
    """
    コンテキストのエンコードを行う。
    """
    os.makedirs(save_dir,exist_ok=True)

    for title,context in tqdm(contexts.items()):
        encoding = tokenizer.encode_plus(
            context,
            return_tensors="pt",
            add_special_tokens=True,
            padding="max_length",
            return_attention_mask=True,
            max_length=max_seq_length,
            truncation=True
        )

        input_ids=encoding["input_ids"].view(-1)

        title_hash=hashing.get_md5_hash(title)
        save_filepath=os.path.join(save_dir,title_hash+".pt")
        torch.save(input_ids,save_filepath)

def main(context_filepath:str,save_dir:str):
    logger.info("Tokenizerを作成します。")
    tokenizer=BertJapaneseTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")

    logger.info("{}からコンテキストを読み込みます。".format(context_filepath))
    contexts=load_contexts(context_filepath)

    logger.info("エンコードを開始します。")
    encode_contexts(tokenizer,contexts,512,save_dir)
    logger.info("エンコードを終了しました。")

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--context_filepath",type=str)
    parser.add_argument("--save_dir",type=str)
    args=parser.parse_args()

    main(
        args.context_filepath,
        args.save_dir
    )
