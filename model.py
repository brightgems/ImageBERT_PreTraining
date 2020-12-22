import random
import torch
import torch.nn as nn
from transformers import(
    BertConfig,
    BertTokenizer,
    BertPreTrainedModel
)
from typing import Tuple

from imagebert.model import ImageBertModel,BERT_MAX_SEQ_LENGTH

class ImageBertForPreTraining(BertPreTrainedModel):
    """
    ImageBERTのPre-Trainingを行うためのクラス
    """
    def __init__(
        self,
        config:BertConfig,
        roi_features_dim:int=1024,
        num_classes:int=80):
        super().__init__(config)

        self.imbert=ImageBertModel(config)
        self.fc_mlm=nn.Linear(config.hidden_size,config.vocab_size)
        self.fc_moc=nn.Linear(config.hidden_size,num_classes)
        self.fc_mrfr=nn.Linear(config.hidden_size,roi_features_dim)
        self.fc_itm=nn.Linear(config.hidden_size,1)

        self.init_weights()

        #setup_image_bert()でモデルをセットアップするか
        #set_mask_token_id()で明示的に設定すると有効になる。
        self.mask_token_id=None

    def to(self,device:torch.device):
        super().to(device)

        self.imbert.to(device)
        self.fc_mlm.to(device)
        self.fc_moc.to(device)
        self.fc_mrfr.to(device)
        self.fc_itm.to(device)

    def setup_image_bert(self,pretrained_model_name_or_path,*model_args,**kwargs):
        """
        パラメータを事前学習済みのモデルから読み込んでImageBERTのモデルを作成する。
        """
        self.imbert=ImageBertModel.create_from_pretrained(pretrained_model_name_or_path,*model_args,**kwargs)

        tokenizer=BertTokenizer.from_pretrained(pretrained_model_name_or_path)
        self.mask_token_id=tokenizer.mask_token_id

    def set_mask_token_id(self,mask_token_id:int):
        self.mask_token_id=mask_token_id

    def __create_masked_input_ids(
        self,
        input_ids:torch.Tensor)->Tuple[torch.Tensor,torch.Tensor]:
        batch_size=input_ids.size(0)

        masked_input_ids=input_ids.detach().clone().cpu()

        #全体の15%のトークンがマスクされる。
        mask_rnds=torch.rand(batch_size,BERT_MAX_SEQ_LENGTH)
        mask_flags=(mask_rnds<0.15)

        for i in range(batch_size):
            for j in range(BERT_MAX_SEQ_LENGTH):
                if mask_flags[i,j]==False:
                    continue

                prob_rnd=random.random()
                if prob_rnd<0.1:
                    pass
                #10%の確率でランダムなトークンに変更する。
                elif prob_rnd<0.2:
                    masked_input_ids[i,j]=random.randrange(self.config.vocab_size)
                #80%の確率で[MASK]トークンに変更する。
                else:
                    masked_input_ids[i,j]=self.mask_token_id

        return masked_input_ids,mask_flags
    
    def __create_masked_roi_labels(
        self,
        roi_labels:torch.Tensor)->Tuple[torch.Tensor,torch.Tensor]:
        batch_size=roi_labels.size(0)
        max_num_rois=roi_labels.size(1)

        masked_roi_labels=roi_labels.detach().clone().cpu()

        #全体の15%のトークンがマスクされる。
        mask_rnds=torch.rand(batch_size,max_num_rois)
        mask_flags=(mask_rnds<0.15)
        for i in range(batch_size):
            for j in range(max_num_rois):
                if mask_flags[i,j]==False:
                    continue

                prob_rnd=random.random()
                #10%の確率で変更しない。
                if prob_rnd<0.1:
                    pass
                #90%の確率で0に変更する。
                else:
                    masked_roi_labels[i,j]=0

        return masked_roi_labels,mask_flags

    def __create_text_attention_mask(
        self,
        input_ids:torch.Tensor)->torch.Tensor:
        return (input_ids!=0).long().cpu()

    def __create_roi_attention_mask(
        self,
        roi_boxes:torch.Tensor)->torch.Tensor:
        batch_size=roi_boxes.size(0)
        max_num_rois=roi_boxes.size(1)

        roi_attention_mask=torch.empty(batch_size,max_num_rois,dtype=torch.long)
        for i in range(batch_size):
            for j in range(max_num_rois):
                roi_box=roi_boxes[i,j]  #(4)

                #0ベクトルならそのRoIは存在しないので、attention_mask=0
                if torch.all(roi_box<1.0e-8):
                    roi_attention_mask[i,j]=0
                else:
                    roi_attention_mask[i,j]=1

        return roi_attention_mask

    def forward(
        self,
        input_ids:torch.Tensor, #(N,BERT_MAX_SEQ_LENGTH)
        roi_boxes:torch.Tensor,    #(N,max_num_rois,4)
        roi_features:torch.Tensor,  #(N,max_num_rois,roi_features_dim)
        roi_labels:torch.Tensor,    #(N,max_num_rois)
        itm_labels:torch.Tensor,    #(N)    0:負例 1:正例
        output_hidden_states:bool=None,
        use_roi_seq_position:bool=False):
        device=self.fc_mlm.weight.device

        batch_size=roi_boxes.size(0)
        max_num_rois=roi_boxes.size(1)

        masked_input_ids,input_ids_mask_flags=self.__create_masked_input_ids(input_ids)
        masked_input_ids=masked_input_ids[:,:BERT_MAX_SEQ_LENGTH-max_num_rois]
        input_ids_mask_flags=input_ids_mask_flags[:,:BERT_MAX_SEQ_LENGTH-max_num_rois]
        input_ids_attention_mask=self.__create_text_attention_mask(input_ids)
        input_ids_attention_mask=input_ids_attention_mask[:,:BERT_MAX_SEQ_LENGTH-max_num_rois]

        masked_roi_labels,roi_mask_flags=self.__create_masked_roi_labels(roi_labels)
        roi_attention_mask=self.__create_roi_attention_mask(roi_boxes)

        masked_input_tokens=torch.cat([masked_input_ids,masked_roi_labels],dim=1)
        mask_flags=torch.cat([input_ids_mask_flags,roi_mask_flags],dim=1)
        input_attention_mask=torch.cat([input_ids_attention_mask,roi_attention_mask],dim=1)

        masked_input_tokens=masked_input_tokens.to(device)
        input_attention_mask=input_attention_mask.to(device)
        roi_boxes=roi_boxes.to(device)
        roi_features=roi_features.to(device)
        roi_labels=roi_labels.to(device)
        itm_labels=itm_labels.to(device)

        #Forward
        outputs=self.imbert(
            input_ids=masked_input_tokens,
            attention_mask=input_attention_mask,
            roi_boxes=roi_boxes,
            roi_features=roi_features,
            output_hidden_states=output_hidden_states,
            return_dict=False,
            use_roi_seq_position=use_roi_seq_position
        )
        sequence_output,pooled_output=outputs[:2]

        #各種Lossの計算
        criterion_ce=nn.CrossEntropyLoss()
        criterion_mse=nn.MSELoss()
        criterion_bce=nn.BCELoss()

        mlm_loss=0
        moc_loss=0
        mrfr_loss=0
        itm_loss=0
        for i in range(batch_size):
            #MLM, MOC, MRFRの損失は正例についてのみ計算する。
            if itm_labels[i]==0:
                continue

            #Masked Language Modeling (MLM)
            for j in range(BERT_MAX_SEQ_LENGTH-max_num_rois):
                if mask_flags[i,j]:
                    vec=sequence_output[i,j]
                    vec=torch.unsqueeze(vec,0)
                    vec=self.fc_mlm(vec)
                    mlm_loss+=criterion_ce(vec,input_ids[i,j])

            #Masked Object Classification (MOC)
            #Masked Region Feature Regression (MRFR)
            for j in range(BERT_MAX_SEQ_LENGTH-max_num_rois,BERT_MAX_SEQ_LENGTH):
                if mask_flags[i,j]:
                    vec_orig=sequence_output[i,j]
                    vec_orig=torch.unsqueeze(vec_orig,0)
                    vec_moc=self.fc_moc(vec_orig)
                    moc_loss+=criterion_ce(vec_moc,roi_labels[i,j-(BERT_MAX_SEQ_LENGTH-max_num_rois)])

                    vec_mrfr=self.fc_mrfr(vec_orig)
                    mrfr_loss+=criterion_mse(vec_mrfr,roi_features[i,j-(BERT_MAX_SEQ_LENGTH-max_num_rois)])

        #Image-Text Matching (ITM)
        vec=self.fc_itm(pooled_output)
        target=torch.unsqueeze(itm_labels,1)
        itm_loss+=criterion_bce(vec,target)

        total_loss=mlm_loss+moc_loss+mrfr_loss+itm_loss

        output=(sequence_output,pooled_output)+outputs[2:]
        return total_loss,output
