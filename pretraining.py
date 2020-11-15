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

import imagebert.utils as imbutils

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

def load_roi_info_from_files(
    roi_boxes_filepaths:List[str],
    roi_features_filepaths:List[str],
    roi_labels_filepaths:List[str],
    max_num_rois:int,
    roi_features_dim:int):
    """
    RoI情報をファイルから読み込む。
    """
    batch_size=len(roi_boxes_filepaths)
    if len(roi_features_filepaths)!=batch_size or len(roi_labels_filepaths)!=batch_size:
        raise RuntimeError("リストに含まれる要素数が不正です。")

    ret_roi_boxes=torch.empty(batch_size,max_num_rois,4)
    ret_roi_features=torch.empty(batch_size,max_num_rois,roi_features_dim)
    ret_roi_labels=torch.empty(batch_size,max_num_rois,dtype=torch.long)

    for i in range(batch_size):
        roi_boxes=None
        roi_features=None
        roi_labels=None
        #RoIの情報が存在する場合はそれを読み込む。
        if os.path.exists(roi_boxes_filepaths[i]):
            roi_boxes=imbutils.load_roi_boxes_from_file(roi_boxes_filepaths[i],max_num_rois)
            roi_features=imbutils.load_roi_features_from_file(roi_features_filepaths[i],max_num_rois)
            roi_labels=imbutils.load_roi_labels_from_file(roi_labels_filepaths[i],max_num_rois)
        else:
            roi_boxes=torch.zeros(max_num_rois,4)
            roi_features=torch.zeros(max_num_rois,roi_features_dim)
            roi_labels=torch.zeros(max_num_rois,dtype=torch.long)
        
        ret_roi_boxes[i]=roi_boxes
        ret_roi_features[i]=roi_features
        ret_roi_labels[i]=roi_labels

    ret={
        "roi_boxes":ret_roi_boxes,
        "roi_features":ret_roi_features,
        "roi_labels":ret_roi_labels
    }
    return ret

def train(
    im_bert:ImageBertForPreTraining,
    dataloader:DataLoader,
    optimizer:torch.optim.Optimizer,
    max_num_rois:int,
    roi_features_dim:int,
    create_negative_prob:float,
    logging_steps:int=100)->float:
    im_bert.train()

    count_steps=0
    total_loss=0

    for batch_idx,batch in enumerate(dataloader):
        roi_info=load_roi_info_from_files(
            batch["roi_boxes_filepath"],
            batch["roi_features_filepath"],
            batch["roi_labels_filepath"],
            max_num_rois,
            roi_features_dim,
        )

        inputs={
            "input_ids":batch["input_ids"].to(device),
            "roi_boxes":roi_info["roi_boxes"].to(device),
            "roi_features":roi_info["roi_features"].to(device),
            "roi_labels":roi_info["roi_labels"].to(device),
            "create_negative_prob":create_negative_prob,
            "return_dict":True
        }

        #Initialize gradiants
        im_bert.zero_grad()
        #Forward propagation
        outputs=im_bert(**inputs)
        loss=outputs["loss"]
        #Backward propagation
        loss.mean().backward()
        nn.utils.clip_grad_norm_(im_bert.parameters(), 1.0)
        # Update parameters
        optimizer.step()

        count_steps+=1
        total_loss+=loss.mean().item()

        if batch_idx%logging_steps==0:
            logger.info("Step: {}\tLoss: {}".format(batch_idx,loss.mean().item()))

    return total_loss/count_steps

def main(args):
    context_dir:str=args.context_dir
    roi_boxes_dir:str=args.roi_boxes_dir
    roi_features_dir:str=args.roi_features_dir
    roi_labels_dir:str=args.roi_labels_dir
    max_num_rois:int=args.max_num_rois
    roi_features_dim:int=args.roi_features_dim
    batch_size:int=args.batch_size
    num_epochs:int=args.num_epochs
    lr:float=args.lr
    create_negative_prob:float=args.create_negative_prob
    result_save_dir:str=args.result_save_dir
    resume_epoch:int=args.resume_epoch
    use_multi_gpus:bool=args.use_multi_gpus

    logger.info("context_dir: {}".format(context_dir))
    logger.info("roi_boxes_dir: {}".format(roi_boxes_dir))
    logger.info("roi_features_dir: {}".format(roi_features_dir))
    logger.info("roi_labels_dir: {}".format(roi_labels_dir))
    logger.info("RoIの最大数: {}".format(max_num_rois))
    logger.info("RoI特徴量の次元: {}".format(roi_features_dim))
    logger.info("バッチサイズ: {}".format(batch_size))
    logger.info("エポック数: {}".format(num_epochs))
    logger.info("学習率: {}".format(lr))
    logger.info("create_negative_prob: {}".format(create_negative_prob))

    logger.info("結果は{}に保存されます。".format(result_save_dir))
    os.makedirs(result_save_dir,exist_ok=True)

    #ImageBERTモデルの作成
    pretrained_model_name="cl-tohoku/bert-base-japanese-whole-word-masking"
    logger.info("{}から事前学習済みの重みを読み込みます。".format(pretrained_model_name))
    config=BertConfig.from_pretrained(pretrained_model_name)
    im_bert=ImageBertForPreTraining(config)
    im_bert.setup_image_bert(pretrained_model_name)
    im_bert.to(device)

    if use_multi_gpus:
        logger.info("複数のGPUを使用して学習を行います。")
        im_bert=nn.DataParallel(im_bert)
        torch.backends.cudnn.benchmark=True

    #学習を再開する場合
    if resume_epoch is not None:
        checkpoint_filepath=os.path.join(result_save_dir,"checkpoint_{}.pt".format(resume_epoch-1))
        if os.path.exists(checkpoint_filepath)==False:
            raise RuntimeError("チェックポイントが存在しません。")

        logger.info("{}からチェックポイントを読み込みます。".format(checkpoint_filepath))

        parameters=torch.load(checkpoint_filepath,map_location=device)
        im_bert.load_state_dict(parameters)

    #データセットとデータローダの作成
    logger.info("データセットを作成します。")
    dataset=create_dataset(context_dir,roi_boxes_dir,roi_features_dir,roi_labels_dir)

    #Optimizerの作成
    optimizer=AdamW(im_bert.parameters(),lr=lr,eps=1e-8)
    
    #訓練ループ
    logger.info("モデルの訓練を開始します。")
    start_epoch=0 if resume_epoch is None else resume_epoch
    for epoch in range(start_epoch,num_epochs):
        logger.info("===== {}/{} =====".format(epoch,num_epochs-1))

        #データローダの作成
        dataloader=DataLoader(dataset,batch_size=batch_size,shuffle=True)

        #訓練
        mean_loss=train(
            im_bert,
            dataloader,
            optimizer,
            max_num_rois,
            roi_features_dim,
            create_negative_prob,
            logging_steps=100
        )
        logger.info("訓練時の平均損失: {}".format(mean_loss))

        #チェックポイントの保存
        checkpoint_filepath=os.path.join(result_save_dir,"checkpoint_{}.pt".format(epoch))
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
    parser.add_argument("--create_negative_prob",type=float)
    parser.add_argument("--result_save_dir",type=str)
    parser.add_argument("--resume_epoch",type=int)
    parser.add_argument("--use_multi_gpus",action="store_true")
    args=parser.parse_args()

    main(args)
