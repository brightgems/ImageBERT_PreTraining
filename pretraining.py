import argparse
import glob
import logging
import os
import torch
import torch.nn as nn
from torch.utils.data import(
    Dataset,
    DataLoader
)
from tqdm import tqdm
from transformers import AdamW,BertConfig
from typing import List

import imagebert.util as ibutil

import sys
sys.path.append(".")
from model import ImageBertForPreTraining

logging_fmt="%(asctime)s %(levelname)s: %(message)s"
logging.basicConfig(format=logging_fmt)
logger=logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ImageBertDataset(Dataset):
    """
    ImageBERTモデルの訓練に必要なデータをまとめたデータセット

    コンテキストのTensorはこのデータセットに保存しておくが、
    RoIの特徴量はサイズが大きすぎるので、実行時にファイルパスを参照して随時読み込む。
    """
    def __init__(self,roi_boxes_dir:str,roi_features_dir:str,roi_labels_dir:str):
        self.input_ids_list:List[torch.Tensor]=[]
        self.roi_boxes_filepaths:List[str]=[]
        self.roi_features_filepaths:List[str]=[]
        self.roi_labels_filepaths:List[str]=[]

        self.roi_boxes_dir=roi_boxes_dir
        self.roi_features_dir=roi_features_dir
        self.roi_labels_dir=roi_labels_dir

    def __len__(self):
        return len(self.input_ids_list)

    def __getitem__(self,index:int):
        """
        指定されたインデックスのデータをDict形式で返す。
        """
        input_ids=self.input_ids_list[index]
        roi_boxes_filepath=self.roi_boxes_filepaths[index]
        roi_features_filepath=self.roi_features_filepaths[index]
        roi_labels_filepath=self.roi_labels_filepaths[index]

        ret={
            "input_ids":input_ids,
            "roi_boxes_filepath":roi_boxes_filepath,
            "roi_features_filepath":roi_features_filepath,
            "roi_labels_filepath":roi_labels_filepath
        }

        return ret

    def append(self,input_ids_filepath:str):
        input_ids=torch.load(input_ids_filepath,map_location=torch.device("cpu"))

        basename=os.path.basename(input_ids_filepath)
        title_hash=os.path.splitext(basename)[0]

        roi_boxes_filepath=os.path.join(self.roi_boxes_dir,title_hash+".pt")
        roi_features_filepath=os.path.join(self.roi_features_dir,title_hash+".pt")
        roi_labels_filepath=os.path.join(self.roi_labels_dir,title_hash+".pt")

        self.input_ids_list.append(input_ids)
        self.roi_boxes_filepaths.append(roi_boxes_filepath)
        self.roi_features_filepaths.append(roi_features_filepath)
        self.roi_labels_filepaths.append(roi_labels_filepath)

def create_dataset(
    context_dir:str,
    roi_boxes_dir:str,
    roi_features_dir:str,
    roi_labels_dir:str,
    num_examples:int=-1)->ImageBertDataset:
    """
    データセットを作成する。
    """
    pathname=os.path.join(context_dir,"*")
    context_files=glob.glob(pathname)
    logger.info("コンテキストの数: {}".format(len(context_files)))

    dataset=ImageBertDataset(roi_boxes_dir,roi_features_dir,roi_labels_dir)
    for i,context_file in tqdm(enumerate(context_files),total=len(context_files)):
        if num_examples>=0 and i>=num_examples:
            break

        dataset.append(context_file)

    return dataset

def train(
    im_bert:ImageBertForPreTraining,
    dataloader:DataLoader,
    optimizer:torch.optim.Optimizer,
    max_num_rois:int,
    roi_features_dim:int,
    logging_steps:int=100):
    im_bert.train()

    for batch_idx,batch in enumerate(dataloader):
        roi_info=ibutil.load_roi_info_from_files(
            batch["roi_boxes_filepath"],
            batch["roi_features_filepath"],
            batch["roi_labels_filepath"],
            max_num_rois,
            roi_features_dim,
            device
        )

        inputs={
            "input_ids":batch["input_ids"].to(device),
            "roi_boxes":roi_info["roi_boxes"].to(device),
            "roi_features":roi_info["roi_features"].to(device),
            "roi_labels":roi_info["roi_labels"].to(device),
            "return_dict":True
        }

        #Initialize gradiants
        im_bert.zero_grad()
        #Forward propagation
        outputs=im_bert(**inputs)
        loss=outputs["loss"]
        #Backward propagation
        loss.backward()
        nn.utils.clip_grad_norm_(im_bert.parameters(), 1.0)
        # Update parameters
        optimizer.step()

def main(
    context_dir:str,
    roi_boxes_dir:str,
    roi_features_dir:str,
    roi_labels_dir:str,
    max_num_rois:int,
    roi_features_dim:int,
    batch_size:int,
    num_epochs:int,
    lr:float,
    result_save_dir:str):
    logger.info("context_dir: {}".format(context_dir))
    logger.info("roi_boxes_dir: {}".format(roi_boxes_dir))
    logger.info("roi_features_dir: {}".format(roi_features_dir))
    logger.info("roi_labels_dir: {}".format(roi_labels_dir))
    logger.info("RoIの最大数: {}".format(max_num_rois))
    logger.info("RoI特徴量の次元: {}".format(roi_features_dim))
    logger.info("バッチサイズ: {}".format(batch_size))
    logger.info("エポック数: {}".format(num_epochs))
    logger.info("学習率: {}".format(lr))

    logger.info("結果は{}に保存されます。".format(result_save_dir))
    os.makedirs(result_save_dir,exist_ok=True)

    #ImageBERTモデルの作成
    pretrained_model_name="cl-tohoku/bert-base-japanese-whole-word-masking"
    logger.info("{}から事前学習済みの重みを読み込みます。".format(pretrained_model_name))
    config=BertConfig.from_pretrained(pretrained_model_name)
    im_bert=ImageBertForPreTraining(config)
    im_bert.setup_image_bert(pretrained_model_name)

    #データセットとデータローダの作成
    logger.info("データセットを作成します。")
    dataset=create_dataset(context_dir,roi_boxes_dir,roi_features_dir,roi_labels_dir,10)

    #Optimizerの作成
    optimizer=AdamW(im_bert.parameters(),lr=lr,eps=1e-8)
    
    #訓練ループ
    logger.info("モデルの訓練を開始します。")
    for epoch in range(num_epochs):
        logger.info("===== {}/{} =====".format(epoch+1,num_epochs))

        #データローダの作成
        dataloader=DataLoader(dataset,batch_size=batch_size,shuffle=True)

        #訓練
        train(
            im_bert,
            dataloader,
            optimizer,
            max_num_rois,
            roi_features_dim,
            logging_steps=100
        )

        #チェックポイントの保存
        checkpoint_filepath=os.path.join(result_save_dir,"checkpoint_{}.pt".format(epoch+1))
        torch.save(im_bert.state_dict(),checkpoint_filepath)

if __name__=="__main__":
    parser=argparse.ArgumentParser()

    parser.add_argument("--context_dir",type=str)
    parser.add_argument("--roi_boxes_dir",type=str)
    parser.add_argument("--roi_features_dir",type=str)
    parser.add_argument("--roi_labels_dir",type=str)
    parser.add_argument("--max_num_rois",type=int)
    parser.add_argument("--roi_features_dim",type=int)
    parser.add_argument("--batch_size",type=int)
    parser.add_argument("--num_epochs",type=int)
    parser.add_argument("--lr",type=float)
    parser.add_argument("--result_save_dir",type=str)
    
    args=parser.parse_args()

    main(
        args.context_dir,
        args.roi_boxes_dir,
        args.roi_features_dir,
        args.roi_labels_dir,
        args.max_num_rois,
        args.roi_features_dim,
        args.batch_size,
        args.num_epochs,
        args.lr,
        args.result_save_dir
    )
