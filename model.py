import random
import torch
import torch.nn as nn
from transformers import(
    BertConfig,
    BertTokenizer,
    BertPreTrainedModel
)
from transformers.modeling_bert import BertPreTrainingHeads
from typing import Tuple

from imagebert.model import ImageBertModel,BERT_MAX_SEQ_LENGTH

class ImageBertForPreTraining(BertPreTrainedModel):
    """
    ImageBERTのPre-Trainingを行うためのクラス
    """
    def __init__(self,config:BertConfig,roi_features_dim:int=1024):
        super().__init__(config)

        self.imbert=ImageBertModel(config)
        self.cls=BertPreTrainingHeads(config)
        self.fc_mrfr=nn.Linear(config.hidden_size,roi_features_dim)

        self.init_weights()

        #setup_image_bert()でモデルをセットアップするか
        #set_mask_token_id()で明示的に設定すると有効になる。
        self.mask_token_id=None

    def to(self,device:torch.device):
        super().to(device)

        self.imbert.to(device)
        self.cls.to(device)
        self.fc_mrfr.to(device)

    def setup_image_bert(self,pretrained_model_name_or_path,*model_args,**kwargs):
        """
        パラメータを事前学習済みのモデルから読み込んでImageBERTのモデルを作成する。
        """
        self.imbert=ImageBertModel.create_from_pretrained(pretrained_model_name_or_path,*model_args,**kwargs)

        tokenizer=BertTokenizer.from_pretrained(pretrained_model_name_or_path)
        self.mask_token_id=tokenizer.mask_token_id

    def set_mask_token_id(self,mask_token_id:int):
        self.mask_token_id=mask_token_id

    def __create_masked_token_ids_and_masked_lm_labels(
        self,
        input_ids:torch.Tensor,
        max_num_rois:int)->Tuple[torch.Tensor,torch.Tensor]:
        """
        返されるTensorはRoI部分を考慮してTruncateされたもの
        """
        device=self.fc_mrfr.weight.device
        batch_size=input_ids.size(0)

        masked_token_ids=input_ids.detach().clone()
        masked_token_ids=masked_token_ids[:,:BERT_MAX_SEQ_LENGTH-max_num_rois]
        masked_lm_labels=torch.ones(batch_size,BERT_MAX_SEQ_LENGTH-max_num_rois,dtype=torch.long)*(-100)
        masked_lm_labels=masked_lm_labels.to(device)

        #全体の15%のトークンがマスクされる。
        mask_rnds=torch.rand(batch_size,BERT_MAX_SEQ_LENGTH-max_num_rois).to(device)
        mask_flags=(mask_rnds<0.15).to(device)
        for i in range(batch_size):
            for j in range(BERT_MAX_SEQ_LENGTH-max_num_rois):
                if mask_flags[i,j]==False:
                    continue

                prob_rnd=random.random()
                #10%の確率で変更しない。
                if prob_rnd<0.1:
                    pass
                #10%の確率でランダムなトークンに変更する。
                elif prob_rnd<0.2:
                    masked_lm_labels[i,j]=input_ids[i,j]
                    masked_token_ids[i,j]=random.randrange(self.config.vocab_size)
                #80%の確率で[MASK]トークンに変更する。
                else:
                    masked_lm_labels[i,j]=input_ids[i,j]
                    masked_token_ids[i,j]=self.mask_token_id

        return masked_token_ids,masked_lm_labels

    def __create_masked_roi_labels_and_masked_oc_labels(
        self,
        roi_labels:torch.Tensor)->Tuple[torch.Tensor,torch.Tensor]:
        device=self.fc_mrfr.weight.device
        batch_size=roi_labels.size(0)
        max_num_rois=roi_labels.size(1)

        masked_roi_labels=roi_labels.detach().clone()
        masked_oc_labels=torch.ones(batch_size,max_num_rois,dtype=torch.long)*(-100)
        masked_oc_labels=masked_oc_labels.to(device)

        #全体の15%のトークンがマスクされる。
        mask_rnds=torch.rand(batch_size,max_num_rois).to(device)
        mask_flags=(mask_rnds<0.15).to(device)
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
                    masked_oc_labels[i,j]=roi_labels[i,j]
                    masked_roi_labels[i,j]=0

        return masked_roi_labels,masked_oc_labels

    def __create_negative_samples(
        self,
        input_ids:torch.Tensor, #(N,BERT_SEQ_MAX_LENGTH)
        roi_boxes:torch.Tensor, #(N,max_num_rois,4)
        roi_features:torch.Tensor,  #(N,max_num_rois,roi_features_dim)
        roi_labels:torch.Tensor,    #(N,max_num_rois)
        create_negative_prob:float):   
        """
        Image-Text Matching (ITM)を行うための負例を作成する。

        create_negative_probで指定された確率でバッチに含まれるサンプルを負例にする。
        たとえばcreate_negative_prob=0.5なら、
        50%の確率でこのバッチの全サンプルはテキストとRoIが対応しない負例になり、
        50%の確率で何も変更されない。

        負例を作成する場合には、n番目のサンプルのRoIデータを(n+1)%N番目のRoIデータに変更する。

        作成された例はDict形式で返される。
        """
        device=self.fc_mrfr.weight.device

        if random.random()>create_negative_prob:
            ret={
                "input_ids":input_ids.to(device),
                "roi_boxes":roi_boxes.to(device),
                "roi_features":roi_features.to(device),
                "roi_labels":roi_labels.to(device),
                "is_negative":False
            }
            return ret

        batch_size=input_ids.size(0)

        sample_0={
            "roi_boxes":roi_boxes[0].detach().clone(),
            "roi_features":roi_features[0].detach().clone(),
            "roi_labels":roi_labels[0].detach().clone()
        }
        for i in range(batch_size):
            if i==batch_size-1:
                roi_boxes[i]=sample_0["roi_boxes"]
                roi_features[i]=sample_0["roi_features"]
                roi_labels[i]=sample_0["roi_labels"]
            else:
                roi_boxes[i]=roi_boxes[i+1]
                roi_features[i]=roi_features[i+1]
                roi_labels[i]=roi_labels[i+1]

        ret={
            "input_ids":input_ids.to(device),
            "roi_boxes":roi_boxes.to(device),
            "roi_features":roi_features.to(device),
            "roi_labels":roi_labels.to(device),
            "is_negative":True
        }
        return ret

    def __create_attention_mask(
        self,
        input_ids:torch.Tensor,
        roi_boxes:torch.Tensor)->torch.Tensor:
        device=self.fc_mrfr.weight.device

        batch_size=roi_boxes.size(0)
        max_num_rois=roi_boxes.size(1)

        roi_attention_mask=torch.empty(batch_size,max_num_rois,dtype=torch.long).to(device)
        for i in range(batch_size):
            for j in range(max_num_rois):
                roi_box=roi_boxes[i,j]  #(4)

                #0ベクトルならそのRoIは存在しないので、attention_mask=0
                if torch.all(roi_box<1.0e-8):
                    roi_attention_mask[i,j]=0
                else:
                    roi_attention_mask[i,j]=1
        
        text_attention_mask=(input_ids!=0).long().to(device)
        text_attention_mask=text_attention_mask[:,:BERT_MAX_SEQ_LENGTH-max_num_rois]
        attention_mask=torch.cat([text_attention_mask,roi_attention_mask],dim=1)

        return attention_mask

    def forward(
        self,
        input_ids:torch.Tensor, #(N,BERT_MAX_SEQ_LENGTH)
        roi_boxes:torch.Tensor,    #(N,max_num_rois,4)
        roi_features:torch.Tensor,  #(N,max_num_rois,roi_features_dim)
        roi_labels:torch.Tensor,    #(N,max_num_rois)
        create_negative_prob:float=0.2,
        output_hidden_states:bool=None,
        return_dict:bool=None,
        use_roi_seq_position:bool=False):
        """
        roi_labelsはFaster R-CNNで検出されたRoIのクラス
        """
        device=self.fc_mrfr.weight.device

        batch_size=roi_boxes.size(0)
        max_num_rois=roi_boxes.size(1)

        #入力サンプルの作成
        samples=self.__create_negative_samples(input_ids,roi_boxes,roi_features,roi_labels,create_negative_prob)
        input_ids=samples["input_ids"]
        roi_boxes=samples["roi_boxes"]
        roi_features=samples["roi_features"]
        roi_labels=samples["roi_labels"]
        is_negative=samples["is_negative"]

        itm_labels=None
        if is_negative:
            itm_labels=torch.ones(batch_size,dtype=torch.long).to(device)
        else:
            itm_labels=torch.zeros(batch_size,dtype=torch.long).to(device)

        #Masked Language Modeling (MLM)およびMasked Object Classification (MOC)用の入力の作成
        masked_token_ids,masked_lm_labels=self.__create_masked_token_ids_and_masked_lm_labels(input_ids,max_num_rois)
        masked_roi_labels,masked_oc_labels=self.__create_masked_roi_labels_and_masked_oc_labels(roi_labels)

        masked_input_ids=torch.cat([masked_token_ids,masked_roi_labels],dim=1)
        masked_lm_oc_labels=torch.cat([masked_lm_labels,masked_oc_labels],dim=1)

        #Attention Maskを作成する。
        attention_mask=self.__create_attention_mask(input_ids,roi_boxes)

        #Attention Maskが0の部分はLoss計算時に無視する。
        masked_lm_oc_labels[attention_mask==0]=-100

        #Forward
        outputs=self.imbert(
            input_ids=masked_input_ids,
            attention_mask=attention_mask,
            roi_boxes=roi_boxes,
            roi_features=roi_features,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_roi_seq_position=use_roi_seq_position
        )

        #各種Lossの計算
        criterion_ce=nn.CrossEntropyLoss()
        criterion_mse=nn.MSELoss()

        #Masked Language Modeling (MLM)
        #Masked Object Classification (MOC)
        #Image-Text Matching (ITM)
        sequence_output,pooled_output=outputs[:2]
        prediction_scores,seq_relationship_score=self.cls(sequence_output,pooled_output)

        masked_lm_oc_loss=0
        if is_negative==False:
            masked_lm_oc_loss=criterion_ce(prediction_scores.view(-1,self.config.vocab_size),masked_lm_oc_labels.view(-1))
        
        itm_loss=criterion_ce(seq_relationship_score.view(-1,2),itm_labels.view(-1))

        #Masked Region Feature Regression (MRFR)
        mrfr_loss=0
        if is_negative==False:
            for i in range(batch_size):
                for j in range(BERT_MAX_SEQ_LENGTH-max_num_rois,BERT_MAX_SEQ_LENGTH):
                    #マスクされているRoIトークンについてLossを計算する。
                    if masked_lm_oc_labels[i,j]!=-100:
                        input=sequence_output[i,j]
                        input=self.fc_mrfr(input)
                        target=roi_features[i,j-(BERT_MAX_SEQ_LENGTH-max_num_rois)]

                        mrfr_loss+=criterion_mse(input,target)

        total_loss=masked_lm_oc_loss+itm_loss+mrfr_loss

        if not return_dict:
            output = (prediction_scores, seq_relationship_score) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        ret={
            "loss":total_loss,
            "prediction_logits":prediction_scores,
            "seq_relationship_logits":seq_relationship_score,
            "hidden_states":outputs.hidden_states,
            "attentions":outputs.attentions
        }
        return ret
