import argparse
import logging
import os
from imagebert.model import BERT_MAX_SEQ_LENGTH
import torch
import torch.nn as nn
from torch.utils.data import(
    Dataset,
    DataLoader
)
from tqdm import tqdm
from transformers import (
    AdamW,
    BertConfig,
    BertJapaneseTokenizer
)
from typing import List

import imagebert.utils as imbutils

import sys
sys.path.append(".")
from model import ImageBertForPreTraining

logging_fmt="%(asctime)s %(levelname)s: %(message)s"
logging.basicConfig(format=logging_fmt)
logger=logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

class PretrainingDataset(Dataset):
    """
    ImageBERTモデルの訓練に必要なデータをまとめたデータセット
    実際のデータではなく、データのファイル名を保存しておく。
    """
    def __init__(
        self,
        input_ids_dir:str,
        roi_boxes_dir:str,
        roi_features_dir:str,
        roi_labels_dir:str):
        self.input_ids_filenames:List[str]=[]
        self.caption_indices:List[int]=[]   #STAIR Captionsで使用される。
        self.roi_filenames:List[str]=[]

        self.input_ids_dir=input_ids_dir
        self.roi_boxes_dir=roi_boxes_dir
        self.roi_features_dir=roi_features_dir
        self.roi_labels_dir=roi_labels_dir

    def __len__(self):
        return len(self.input_ids_filenames)

    def __getitem__(self,index:int):
        """
        読み込むべきファイルへのファイルパスを返す。
        """
        input_ids_filename=self.input_ids_filenames[index]
        caption_index=self.caption_indices[index]
        roi_filename=self.roi_filenames[index]

        is_positive_sample=True if input_ids_filename==roi_filename else False

        input_ids_filepath=os.path.join(self.input_ids_dir,input_ids_filename)
        roi_boxes_filepath=os.path.join(self.roi_boxes_dir,roi_filename)
        roi_features_filepath=os.path.join(self.roi_features_dir,roi_filename)
        roi_labels_filepath=os.path.join(self.roi_labels_dir,roi_filename)

        ret={
            "input_ids_filepath":input_ids_filepath,
            "caption_index":caption_index,
            "roi_boxes_filepath":roi_boxes_filepath,
            "roi_features_filepath":roi_features_filepath,
            "roi_labels_filepath":roi_labels_filepath,
            "is_positive_sample":is_positive_sample
        }
        return ret

    def append(
        self,
        input_ids_filename:str,
        caption_index:int,
        roi_filename:str):
        self.input_ids_filenames.append(input_ids_filename)
        self.caption_indices.append(caption_index)
        self.roi_filenames.append(roi_filename)

def create_dataset(
    sample_list_filepath:str,
    input_ids_dir:str,
    roi_boxes_dir:str,
    roi_features_dir:str,
    roi_labels_dir:str,
    num_samples:int)->PretrainingDataset:
    """
    データセットを作成する。
    """
    dataset=PretrainingDataset(input_ids_dir,roi_boxes_dir,roi_features_dir,roi_labels_dir)

    with open(sample_list_filepath,"r",encoding="utf_8") as r:
        lines=r.read().splitlines()

    lines=lines[:num_samples]
    for line in lines:
        splits=line.split("\t")
        if len(splits)==2:
            input_ids_filename,roi_filename=splits[:2]
            dataset.append(input_ids_filename,0,roi_filename)
        elif len(splits)==3:
            input_ids_filename,caption_index,roi_filename=splits[:3]
            dataset.append(input_ids_filename,caption_index,roi_filename)
        else:
            raise RuntimeError("サンプルリストの形式が不正です。")

    return dataset

def load_input_ids_from_files(
    input_ids_filepaths:List[str],
    caption_indices:List[int]=None)->torch.Tensor:
    """
    テキスト入力(Input IDs)をファイルから読み込む。
    STAIR Captionsを用いる場合には、二つ目の引数にキャプションのインデックスを渡す。
    """
    batch_size=len(input_ids_filepaths)

    ret_input_ids=torch.empty(batch_size,BERT_MAX_SEQ_LENGTH,dtype=torch.long)
    for i in range(batch_size):
        input_ids=torch.load(input_ids_filepaths[i])
        if caption_indices is None:
            ret_input_ids[i]=input_ids
        else:
            caption_index=caption_indices[i]
            ret_input_ids[i]=input_ids[caption_index]

    return ret_input_ids

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

    ret_roi_boxes=torch.empty(batch_size,max_num_rois,4)
    ret_roi_features=torch.empty(batch_size,max_num_rois,roi_features_dim)
    ret_roi_labels=torch.empty(batch_size,max_num_rois,dtype=torch.long)

    for i in range(batch_size):
        roi_boxes=imbutils.load_roi_boxes_from_file(roi_boxes_filepaths[i],max_num_rois)
        roi_features=imbutils.load_roi_features_from_file(roi_features_filepaths[i],max_num_rois)
        roi_labels=imbutils.load_roi_labels_from_file(roi_labels_filepaths[i],max_num_rois)
        
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
    is_stair_captions:bool,
    logging_steps:int=100):
    """
    モデルの訓練を1エポック進める。
    このエポックでの平均損失はDict形式で返される。
    """
    im_bert.train()

    count_steps=0
    accum_total_loss=0
    accum_mlm_loss=0
    accum_moc_loss=0
    accum_mrfr_loss=0
    accum_itm_loss=0

    for batch_idx,batch in enumerate(dataloader):
        input_ids=None
        if is_stair_captions:
            input_ids=load_input_ids_from_files(batch["input_ids_filepath"],batch["caption_index"])
        else:
            input_ids=load_input_ids_from_files(batch["input_ids_filepath"])

        roi_info=load_roi_info_from_files(
            batch["roi_boxes_filepath"],
            batch["roi_features_filepath"],
            batch["roi_labels_filepath"],
            max_num_rois,
            roi_features_dim,
        )

        itm_labels=batch["is_positive_sample"]
        itm_labels=torch.tensor(itm_labels).long()

        inputs={
            "input_ids":input_ids,
            "roi_boxes":roi_info["roi_boxes"],
            "roi_features":roi_info["roi_features"],
            "roi_labels":roi_info["roi_labels"],
            "itm_labels":itm_labels
        }

        #Initialize gradiants
        im_bert.zero_grad()
        #Forward propagation
        outputs=im_bert(**inputs)
        total_loss=outputs["total_loss"]
        #Backward propagation
        total_loss=total_loss.mean()
        total_loss.backward()
        # Update parameters
        optimizer.step()

        count_steps+=1
        accum_total_loss+=total_loss.item()

        #各タスクのロスも記録しておく。
        mlm_loss=outputs["mlm_loss"]
        moc_loss=outputs["moc_loss"]
        mrfr_loss=outputs["mrfr_loss"]
        itm_loss=outputs["itm_loss"]
        mlm_loss=mlm_loss.mean()
        moc_loss=moc_loss.mean()
        mrfr_loss=mrfr_loss.mean()
        itm_loss=itm_loss.mean()
        accum_mlm_loss+=mlm_loss.item()
        accum_moc_loss+=moc_loss.item()
        accum_mrfr_loss+=mrfr_loss.item()
        accum_itm_loss+=itm_loss.item()

        if batch_idx%logging_steps==0:
            logger.info("Step: {}\tTotal: {}\tMLM: {}\tMOC: {}\tMRFR: {}\tITM: {}".format(
                batch_idx,total_loss.item(),mlm_loss.item(),moc_loss.item(),mrfr_loss.item(),itm_loss.item())
            )

    ret={
        "total_loss":accum_total_loss/count_steps,
        "mlm_loss":accum_mlm_loss/count_steps,
        "moc_loss":accum_moc_loss/count_steps,
        "mrfr_loss":accum_mrfr_loss/count_steps,
        "itm_loss":accum_itm_loss/count_steps
    }
    return ret

def main(args):
    sample_list_filepath:str=args.sample_list_filepath
    input_ids_dir:str=args.input_ids_dir
    roi_boxes_dir:str=args.roi_boxes_dir
    roi_features_dir:str=args.roi_features_dir
    roi_labels_dir:str=args.roi_labels_dir
    max_num_rois:int=args.max_num_rois
    roi_features_dim:int=args.roi_features_dim
    batch_size:int=args.batch_size
    num_epochs:int=args.num_epochs
    lr:float=args.lr
    weight_decay:float=args.weight_decay
    result_save_dir:str=args.result_save_dir
    resume_epoch:int=args.resume_epoch
    imbert_checkpoint_filepath:str=args.imbert_checkpoint_filepath
    use_multi_gpus:bool=args.use_multi_gpus
    no_init_params_from_pretrained_bert:bool=args.no_init_params_from_pretrained_bert
    num_samples:int=args.num_samples
    is_stair_captions:bool=args.is_stair_captions

    logger.info("sample_list_filepath: {}".format(sample_list_filepath))
    logger.info("input_ids_dir: {}".format(input_ids_dir))
    logger.info("roi_boxes_dir: {}".format(roi_boxes_dir))
    logger.info("roi_features_dir: {}".format(roi_features_dir))
    logger.info("roi_labels_dir: {}".format(roi_labels_dir))
    logger.info("max_num_rois: {}".format(max_num_rois))
    logger.info("roi_features_dim: {}".format(roi_features_dim))
    logger.info("batch_size: {}".format(batch_size))
    logger.info("num_epochs: {}".format(num_epochs))
    logger.info("lr: {}".format(lr))
    logger.info("weight_decay: {}".format(weight_decay))
    logger.info("num_samples: {}".format(num_samples))

    if os.path.exists(input_ids_dir)==False:
        raise RuntimeError("input_ids_dirが存在しません。")
    if os.path.exists(roi_boxes_dir)==False:
        raise RuntimeError("roi_boxes_dirが存在しません。")
    if os.path.exists(roi_features_dir)==False:
        raise RuntimeError("roi_features_dirが存在しません。")
    if os.path.exists(roi_labels_dir)==False:
        raise RuntimeError("roi_labels_dirが存在しません。")

    logger.info("結果は{}に保存されます。".format(result_save_dir))
    os.makedirs(result_save_dir,exist_ok=True)

    #ImageBERTモデルの作成
    im_bert=None
    pretrained_model_name="cl-tohoku/bert-base-japanese-whole-word-masking"
    config=BertConfig.from_pretrained(pretrained_model_name)
    if no_init_params_from_pretrained_bert:
        im_bert=ImageBertForPreTraining(config)
        tokenizer=BertJapaneseTokenizer.from_pretrained(pretrained_model_name)
        im_bert.set_mask_token_id(tokenizer.mask_token_id)
    else:
        logger.info("{}から事前学習済みの重みを読み込みます。".format(pretrained_model_name))
        im_bert=ImageBertForPreTraining(config)
        im_bert.setup_image_bert(pretrained_model_name)

    if imbert_checkpoint_filepath is not None:
        logger.info("{}からチェックポイントを読み込みます。".format(imbert_checkpoint_filepath))
        parameters=torch.load(imbert_checkpoint_filepath,map_location=torch.device("cpu"))
        im_bert.load_state_dict(parameters)

    #学習を再開する場合
    if resume_epoch is not None:
        checkpoint_filepath=os.path.join(result_save_dir,"checkpoint_{}.pt".format(resume_epoch-1))
        if os.path.exists(checkpoint_filepath)==False:
            raise RuntimeError("チェックポイントが存在しません。")

        logger.info("{}からチェックポイントを読み込みます。".format(checkpoint_filepath))
        parameters=torch.load(checkpoint_filepath,map_location=torch.device("cpu"))
        im_bert.load_state_dict(parameters)

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    im_bert.to(device)

    if use_multi_gpus:
        logger.info("複数のGPUを使用して学習を行います。")
        im_bert=nn.DataParallel(im_bert)
        torch.backends.cudnn.benchmark=True

    #データセットとデータローダの作成
    logger.info("データセットを作成します。")
    dataset=create_dataset(
        sample_list_filepath,
        input_ids_dir,
        roi_boxes_dir,
        roi_features_dir,
        roi_labels_dir,
        num_samples=num_samples
    )
    logger.info("データ数: {}".format(len(dataset)))

    #Optimizerの作成
    optimizer=AdamW(im_bert.parameters(),lr=lr,weight_decay=weight_decay)
    
    #訓練ループ
    logger.info("モデルの訓練を開始します。")
    start_epoch=0 if resume_epoch is None else resume_epoch
    for epoch in range(start_epoch,num_epochs):
        logger.info("===== {}/{} =====".format(epoch,num_epochs-1))

        #データローダの作成
        dataloader=DataLoader(dataset,batch_size=batch_size,shuffle=True)

        #訓練
        train_res=train(
            im_bert,
            dataloader,
            optimizer,
            max_num_rois,
            roi_features_dim,
            is_stair_captions,
            logging_steps=100
        )
        mean_total_loss=train_res["total_loss"]
        mean_mlm_loss=train_res["mlm_loss"]
        mean_moc_loss=train_res["moc_loss"]
        mean_mrfr_loss=train_res["mrfr_loss"]
        mean_itm_loss=train_res["itm_loss"]

        logger.info("Total Loss: {}\tMLM Loss: {}\tMOC Loss: {}\t MRFR Loss: {}\tITM Loss: {}".format(
            mean_total_loss,mean_mlm_loss,mean_moc_loss,mean_mrfr_loss,mean_itm_loss)
        )

        #チェックポイントの保存
        checkpoint_filepath=os.path.join(result_save_dir,"checkpoint_{}.pt".format(epoch))
        if use_multi_gpus:
            torch.save(im_bert.module.state_dict(),checkpoint_filepath)
        else:
            torch.save(im_bert.state_dict(),checkpoint_filepath)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--sample_list_filepath",type=str)
    parser.add_argument("--input_ids_dir",type=str)
    parser.add_argument("--roi_boxes_dir",type=str)
    parser.add_argument("--roi_features_dir",type=str)
    parser.add_argument("--roi_labels_dir",type=str)
    parser.add_argument("--max_num_rois",type=int)
    parser.add_argument("--roi_features_dim",type=int)
    parser.add_argument("--batch_size",type=int)
    parser.add_argument("--num_epochs",type=int)
    parser.add_argument("--lr",type=float)
    parser.add_argument("--weight_decay",type=float)
    parser.add_argument("--result_save_dir",type=str)
    parser.add_argument("--resume_epoch",type=int)
    parser.add_argument("--imbert_checkpoint_filepath",type=str)
    parser.add_argument("--use_multi_gpus",action="store_true")
    parser.add_argument("--no_init_params_from_pretrained_bert",action="store_true")
    parser.add_argument("--num_samples",type=int,default=-1)
    parser.add_argument("--is_stair_captions",action="store_true")
    args=parser.parse_args()

    main(args)
