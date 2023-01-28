import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import transformers
from transformers import RobertaTokenizer
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel, RobertaLMHead
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertLMPredictionHead
from transformers.activations import gelu
from transformers.file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions

class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)

        return x

class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """
    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ["cls", "cls_before_pooler", "avg", "avg_top2", "avg_first_last"], "unrecognized pooling type %s" % self.pooler_type

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states

        if self.pooler_type in ['cls_before_pooler', 'cls']:
            return last_hidden[:, 0]
        elif self.pooler_type == "avg":
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError


def cl_init(cls, config):
    """
    Contrastive learning class init function.
    """
    cls.pooler_type = cls.model_args.pooler_type
    cls.pooler = Pooler(cls.model_args.pooler_type)
    if cls.model_args.pooler_type == "cls":
        cls.mlp = MLPLayer(config)
    cls.sim = Similarity(temp=cls.model_args.temp)
    cls.init_weights()

def cl_forward(cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    mlm_input_ids=None,
    mlm_labels=None,
):
    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict
    batch_size = input_ids.size(0)
    # Number of sentences in one instance
    # 2: pair instance; 3: pair instance with a hard negative
    num_sent = input_ids.size(1)

    # Flatten input for encoding
    input_ids = input_ids.view((-1, input_ids.size(-1))) # (bs * num_sent, len)
    attention_mask = attention_mask.view((-1, attention_mask.size(-1))) # (bs * num_sent len)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1))) # (bs * num_sent, len)

    # Get raw embeddings
    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
        return_dict=True,
    )

    # Pooling
    pooler_output = cls.pooler(attention_mask, outputs)
    pooler_output = pooler_output.view((batch_size, num_sent, pooler_output.size(-1))) # (bs, num_sent, hidden)

    # If using "cls", we add an extra MLP layer
    # (same as BERT's original implementation) over the representation.
    if cls.pooler_type == "cls":
        pooler_output = cls.mlp(pooler_output)

    # Separate representation
    z1, z2 = pooler_output[:, 0], pooler_output[:, 1]

    # Gather all embeddings if using distributed training
    if dist.is_initialized() and cls.training:
        # Dummy vectors for allgather
        z1_list = [torch.zeros_like(z1) for _ in range(dist.get_world_size())]
        z2_list = [torch.zeros_like(z2) for _ in range(dist.get_world_size())]
        # Allgather
        dist.all_gather(tensor_list=z1_list, tensor=z1.contiguous())
        dist.all_gather(tensor_list=z2_list, tensor=z2.contiguous())

        # Since allgather results do not have gradients, we replace the
        # current process's corresponding embeddings with original tensors
        z1_list[dist.get_rank()] = z1
        z2_list[dist.get_rank()] = z2
        # Get full batch embeddings: (bs x N, hidden)
        z1 = torch.cat(z1_list, 0)
        z2 = torch.cat(z2_list, 0)
    cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))
    loss_fct = nn.CrossEntropyLoss()
    # labels = torch.arange(cos_sim.size(0)).long().to(cls.device)
    cos = nn.CosineSimilarity(dim=-1)
    cos_sim_fix = cos(z1.unsqueeze(1), z2.unsqueeze(0))

    if cls.model_args.screen_strategy == "Strategy_1":
        cos_sim_nondiag = cos_sim_fix - torch.diag_embed(torch.diag(cos_sim_fix))
        threshold_weights = torch.where(cos_sim_nondiag > cls.model_args.phi, 1, 0)
        threshold_weights = threshold_weights.to(cls.device) + torch.eye(cos_sim_fix.size(0), device=cls.device)

        _, topkind = torch.topk(cos_sim_nondiag, cls.model_args.topk, dim=1, largest=True)  # maximum value
        topkind_expand = torch.eye(cos_sim.size(0))[topkind]
        topkind_expand = topkind_expand.to(cls.device)
        topkind_false = torch.zeros_like(cos_sim, device=cls.device)
        for i in range(cos_sim.size(0)):
            for j in range(cls.model_args.topk):
                topkind_false[i] += topkind_expand[i][j]
        topkind_false = topkind_false + torch.eye(cos_sim_fix.size(0), device=cls.device)
        topk_threshold = threshold_weights * topkind_false
    elif cls.model_args.screen_strategy == "Strategy_2":
        cos_sim_diag = torch.diag(cos_sim_fix).unsqueeze(1) * torch.ones(1, cos_sim_fix.size(0), device=cls.device)
        cos_sim_diff = torch.abs(cos_sim_diag - cos_sim_fix)
        threshold_weights = torch.where(cos_sim_diff < cls.model_args.phi, 1, 0)
        threshold_weights = threshold_weights.to(cls.device)

        cos_sim_diff = cos_sim_diff + torch.eye(cos_sim_fix.size(0), device=cls.device)
        _, topkind = torch.topk(cos_sim_diff, cls.model_args.topk, dim=1, largest=False)  # minimum value
        topkind_expand = torch.eye(cos_sim.size(0))[topkind]
        topkind_expand = topkind_expand.to(cls.device)
        topkind_false = torch.zeros_like(cos_sim_fix, device=cls.device)
        for i in range(cos_sim.size(0)):
            for j in range(cls.model_args.topk):
                topkind_false[i] += topkind_expand[i][j]
        topkind_false = topkind_false + torch.eye(cos_sim_fix.size(0), device=cls.device)
        topk_threshold = threshold_weights * topkind_false

    cos_sim_diag = torch.exp(cos_sim_fix) - torch.diag_embed(torch.diag(torch.exp(cos_sim_fix)))
    if cls.model_args.loss_strategy == "adaptive_elimination":
        cos_sim_weight = torch.ones_like(cos_sim_fix, device=cls.device) - cos_sim_diag / (
            torch.sum(cos_sim_diag, dim=1).unsqueeze(1))
        false_negative_weight = topk_threshold * cos_sim_weight
        non_false_negative_weight = torch.ones_like(cos_sim_fix, device=cls.device) - topk_threshold
        topk_threshold_dynamic = false_negative_weight + non_false_negative_weight
        cos_sim = cos_sim * topk_threshold_dynamic
        labels = torch.arange(cos_sim.size(0)).long().to(cls.device)
        loss = loss_fct(cos_sim, labels)
    elif cls.model_args.loss_strategy == "adaptive_attraction":
        cos_sim_weight = cos_sim_diag / (torch.sum(cos_sim_diag, dim=1).unsqueeze(1)) + torch.eye(
            cos_sim_fix.size(0), device=cls.device)
        topk_threshold_weight = topk_threshold * cos_sim_weight
        softmax = nn.Softmax(dim=1)
        cos_sim_softmax = softmax(cos_sim)
        cos_sim_softmax_log = torch.log(cos_sim_softmax)
        loss_attract = -(topk_threshold_weight * cos_sim_softmax_log)
        loss_attract_sum = torch.sum(loss_attract, dim=1)
        topk_threshold_sum = torch.sum(topk_threshold, dim=1)
        loss_single = torch.div(loss_attract_sum, topk_threshold_sum)
        loss = torch.mean(loss_single)
    elif cls.model_args.loss_strategy == "elimination":
        non_false_negative_weight = torch.ones_like(cos_sim_fix, device=cls.device) - topk_threshold
        topk_threshold_elim = torch.eye(cos_sim_fix.size(0), device=cls.device) + non_false_negative_weight
        cos_sim = cos_sim * topk_threshold_elim
        labels = torch.arange(cos_sim.size(0)).long().to(cls.device)
        loss = loss_fct(cos_sim, labels)
    elif cls.model_args.loss_strategy == "attraction":
        softmax = nn.Softmax(dim=1)
        cos_sim_softmax = softmax(cos_sim)
        cos_sim_softmax_log = torch.log(cos_sim_softmax)
        loss_attract = -(topk_threshold * cos_sim_softmax_log)
        loss_attract_sum = torch.sum(loss_attract, dim=1)
        topk_threshold_sum = torch.sum(topk_threshold, dim=1)
        loss_single = torch.div(loss_attract_sum, topk_threshold_sum)
        loss = torch.mean(loss_single)
   

    if not return_dict:
        output = (cos_sim,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output
    return SequenceClassifierOutput(
        loss=loss,
        logits=cos_sim,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )

def sentemb_forward(
    cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
):

    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict

    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.pooler_type in ['avg_top2', 'avg_first_last'] else False,
        return_dict=True,
    )

    pooler_output = cls.pooler(attention_mask, outputs)
    if cls.pooler_type == "cls" and not cls.model_args.mlp_only_train:
        pooler_output = cls.mlp(pooler_output)

    if not return_dict:
        return (outputs[0], pooler_output) + outputs[2:]

    return BaseModelOutputWithPoolingAndCrossAttentions(
        pooler_output=pooler_output,
        last_hidden_state=outputs.last_hidden_state,
        hidden_states=outputs.hidden_states,
    )


class BertForCL(BertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.bert = BertModel(config, add_pooling_layer=False)

        if self.model_args.do_mlm:
            self.lm_head = BertLMPredictionHead(config)

        cl_init(self, config)

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        mlm_input_ids=None,
        mlm_labels=None,
    ):
        if sent_emb:
            return sentemb_forward(self, self.bert,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            return cl_forward(self, self.bert,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels,
            )


class RobertaForCL(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.roberta = RobertaModel(config, add_pooling_layer=False)

        if self.model_args.do_mlm:
            self.lm_head = RobertaLMHead(config)

        cl_init(self, config)

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        mlm_input_ids=None,
        mlm_labels=None,
    ):
        if sent_emb:
            return sentemb_forward(self, self.roberta,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            return cl_forward(self, self.roberta,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels,
            )
