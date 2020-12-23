import argparse
import glob
import logging
import os
import torch
from tqdm import tqdm
from transformers import BertJapaneseTokenizer
from typing import Dict,List

logging_fmt="%(asctime)s %(levelname)s: %(message)s"
logging.basicConfig(format=logging_fmt)
logger=logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

def load_captions(caption_dir:str)->Dict[str,List[str]]:
    pathname=os.path.join(caption_dir,"*.txt")
    files=glob.glob(pathname)

    captions_dict={}
    for idx,file in enumerate(files):
        with open(file,"r",encoding="utf_8") as r:
            captions=r.read().splitlines()
            filename=os.path.basename(file)
            captions_dict[filename]=captions

            if idx<5:
                logger.info("===== {} =====".format(idx))
                for caption in captions:
                    logger.info("{}".format(caption))

    return captions_dict

def encode_captions(
    tokenizer:BertJapaneseTokenizer,
    captions_dict:Dict[str,List[str]],
    max_seq_length:int,
    save_dir:str):
    os.makedirs(save_dir,exist_ok=True)

    for filename,captions in tqdm(captions_dict.items()):
        input_ids=torch.empty(0,max_seq_length,dtype=torch.long)
        for caption in captions:
            encoding = tokenizer.encode_plus(
                caption,
                return_tensors="pt",
                add_special_tokens=True,
                padding="max_length",
                return_attention_mask=True,
                max_length=max_seq_length,
                truncation=True
            )
            input_ids_tmp=encoding["input_ids"]
            input_ids=torch.cat([input_ids,input_ids_tmp],dim=0)

        save_filename=os.path.splitext(filename)[0]+".pt"
        save_filepath=os.path.join(save_dir,save_filename)
        torch.save(input_ids,save_filepath)

def main(caption_dir:str,save_dir:str):
    logger.info("Tokenizerを作成します。")
    tokenizer=BertJapaneseTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")

    logger.info("{}からキャプションを読み込みます。".format(caption_dir))
    captions_dict=load_captions(caption_dir)

    logger.info("エンコードを開始します。")
    encode_captions(tokenizer,captions_dict,512,save_dir)
    logger.info("エンコードを終了しました。")

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--caption_dir",type=str)
    parser.add_argument("--save_dir",type=str)
    args=parser.parse_args()

    main(args.caption_dir,args.save_dir)
