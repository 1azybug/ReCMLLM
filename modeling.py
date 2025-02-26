from transformers import AutoModelForCausalLM, AutoTokenizer,BitsAndBytesConfig
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaRMSNorm
from transformers.modeling_outputs import CausalLMOutputWithPast
import torch.nn.functional as F
import math
import transformers
import xformers.ops as xops
from transformers.modeling_outputs import BaseModelOutputWithPast
import warnings
from transformers import PreTrainedModel, PretrainedConfig, AutoConfig, AutoModel
import json
from cache import MemCache



def get_uniform_pids(begin, end, ratio, device):
    # [begin-0.5,end+0.5]
    # first range:[begin-0.5,begin-0.5+ratio] or [begin,begin+ratio-1]
    return torch.arange((begin-0.5)+(ratio/2), end+0.5, step=ratio, device=device).unsqueeze(0)



# def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
#     """
#     This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep) [神金]. The hidden states go from (batch,
#     num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
#     """
#     batch, num_key_value_heads, slen, head_dim = hidden_states.shape
#     if n_rep == 1:
#         return hidden_states
#     hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
#     return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, q_cos, q_sin, k_cos, k_sin, position_ids=None, unsqueeze_dim=1):
    # q:[batch_size,q_len,num_q_head,head_dim]
    # k:[batch_size,k_len,num_k_head,head_dim]
    # q_sin/q_cos: [batch_size,q_len,head_dim]
    # k_sin/k_cos: [batch_size,k_len,head_dim]
    q_cos = q_cos.unsqueeze(unsqueeze_dim)
    q_sin = q_sin.unsqueeze(unsqueeze_dim)
    k_cos = k_cos.unsqueeze(unsqueeze_dim)
    k_sin = k_sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * q_cos) + (rotate_half(q) * q_sin)
    k_embed = (k * k_cos) + (rotate_half(k) * k_sin)
    return q_embed, k_embed

    

def LlamaAttention_forward(
    self,
    hidden_states,
    past_key_value = None,
    q_position_embeddings = None,
    kv_position_embeddings = None,
    token_type = None,
):

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states, token_type=token_type)
    key_states = self.k_proj(hidden_states, token_type=token_type)
    value_states = self.v_proj(hidden_states, token_type=token_type)


    # xformer don't require transpose q_len dim and num_head dim.
    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)


    # Training phase only cache the latest_group KVcache; cache the unrotated kv
    if self.training:
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx)
    else:
        raise NotImplementedError("make sure have implemented the inference code for cache.")

    q_cos, q_sin = q_position_embeddings
    k_cos, k_sin = kv_position_embeddings
    print(f"q_cos:{q_cos.shape},q_sin:{q_sin.shape}")
    print(f"k_cos:{k_cos.shape},k_sin:{k_sin.shape}")
    print(f"query_states:{query_states.shape}")
    print(f"key_states:{key_states.shape}")
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, q_cos, q_sin, k_cos, k_sin, unsqueeze_dim=2)

    print(f"self.num_key_value_groups:{self.num_key_value_groups}")
    print(f"0 key_states:{key_states.shape}")
    # key_states = repeat_kv(key_states, self.num_key_value_groups)  
    # value_states = repeat_kv(value_states, self.num_key_value_groups)
    key_states = torch.repeat_interleave(key_states, dim=2, repeats=self.num_key_value_groups)
    value_states = torch.repeat_interleave(value_states, dim=2, repeats=self.num_key_value_groups)
    print(f"1 key_states:{key_states.shape}")
    # Input tensors must be in format [batch size, sequence length, number of heads, embeding size]
    assert query_states.size() == (bsz, q_len, self.num_heads, self.head_dim), "Input tensors must be in format [B, M, H, K], where B is the batch size, M the sequence length, H the number of heads, and K the embeding size per head"
    # print(f"query_states shape:{query_states.shape}, key_states shape:{key_states.shape}")
    attn_output = xops.memory_efficient_attention(
        query_states, key_states, value_states,
        # scale=1.0/math.sqrt(self.head_dim),  # default value is q.shape[-1]**-0.5
        attn_bias=xops.fmha.attn_bias.LowerTriangularFromBottomRightMask()
    )

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)


    attn_output = self.o_proj(attn_output, token_type=token_type)

    return attn_output

def LlamaMLP_forward(self, x, token_type=None):

    down_proj = self.down_proj(self.act_fn(self.gate_proj(x, token_type=token_type)) * self.up_proj(x, token_type=token_type), token_type=token_type)

    return down_proj


def LlamaDecoderLayer_forward(
    self,
    hidden_states,
    past_key_value = None,
    q_position_embeddings = None,
    kv_position_embeddings = None,
    token_type=None, 
):

    residual = hidden_states

    hidden_states = self.input_layernorm(hidden_states, token_type=token_type)

    # Self Attention
    hidden_states = self.self_attn(
        hidden_states=hidden_states,
        past_key_value=past_key_value,
        q_position_embeddings=q_position_embeddings,
        kv_position_embeddings = kv_position_embeddings,
        token_type=token_type,
    )
    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states, token_type=token_type)
    hidden_states = self.mlp(hidden_states, token_type=token_type)
    hidden_states = residual + hidden_states

    return hidden_states


def LlamaModel_forward(
    self,
    position_ids = None,
    past_key_values = None,
    inputs_embeds = None,
    token_type = None,
):

    hidden_states = inputs_embeds

    # create position embeddings to be shared across the decoder layers
    # position_ids:[1,q_len]
    q_position_embeddings = self.rotary_emb(hidden_states, position_ids)

    if past_key_values.kv_cache_is_empty():
        kv_position_embeddings = q_position_embeddings
    else:
        kv_position_ids = torch.cat([past_key_values.get_kvcache_pids(), position_ids], dim=1)
        kv_position_embeddings = self.rotary_emb(hidden_states, kv_position_ids)


    for decoder_layer in self.layers:

        hidden_states = decoder_layer(
            hidden_states,
            past_key_value=past_key_values,
            q_position_embeddings=q_position_embeddings,
            kv_position_embeddings = kv_position_embeddings,
            token_type = token_type
        )   

    hidden_states = self.norm(hidden_states, token_type=token_type)

    return hidden_states



def LlamaForCausalLM_forward(
        self,
        position_ids = None,
        past_key_values = None,
        inputs_embeds = None,
        token_type = None,
    ):
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        hidden_states = self.model(
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            token_type = token_type,
        )

        return hidden_states


class LinearLoraLayer(nn.Module):
    # No bias in LLama3 LinearLayer
    def __init__(self, in_features, out_features, r=256, weight=None):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.scale = 2  # The alpha value is usually twice the rank
        self.lora_A = nn.Parameter(torch.zeros((in_features, r), device=self.weight.device, dtype=torch.bfloat16), requires_grad=True)
        self.lora_B = nn.Parameter(torch.zeros((r, out_features), device=self.weight.device, dtype=torch.bfloat16), requires_grad=True)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        result = F.linear(x, self.weight)
        result += self.scale*(x@self.lora_A@self.lora_B)
        return result


class MoeLinearLoraLayer(nn.Module):
    # No bias in LLama3 LinearLayer
    def __init__(self, in_features, out_features, r=256, weight_base=None, num_experts=1):
        super().__init__()
        self.num_experts = num_experts
        self.weight_base = nn.Parameter(weight_base, requires_grad=False)
        self.scale = 2  # The alpha value is usually twice the rank
        self.lora_A = nn.ParameterList(
            [nn.Parameter(torch.zeros((in_features, r), device=self.weight_base.device, dtype=torch.bfloat16), requires_grad=True) for e in range(num_experts)]
        )
        self.lora_B = nn.ParameterList(
            [nn.Parameter(torch.zeros((r, out_features), device=self.weight_base.device, dtype=torch.bfloat16), requires_grad=True) for e in range(num_experts)]
        )

        for e in range(self.num_experts):
            nn.init.kaiming_uniform_(self.lora_A[e], a=math.sqrt(5))
            nn.init.zeros_(self.lora_B[e])

    def forward(self, x, token_type):
        result_base = F.linear(x, self.weight_base)
        results = torch.zeros_like(result_base)

        for expert in range(self.num_experts):
            # x:[B,S,...]
            expert_mask = (token_type==expert)
            results[:,expert_mask,...] = result_base[:,expert_mask,...] + self.scale*(x[:,expert_mask,...]@self.lora_A[expert]@self.lora_B[expert])

        return results
    

class EmbeddingLoraLayer(nn.Module):
    # No bias in LLama3 LinearLayer
    def __init__(self, in_features, out_features, padding_idx, r=256, weight=None):
        super().__init__()
        self.padding_idx = padding_idx
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.scale = 2  # The alpha value is usually twice the rank
        self.lora_A = nn.Parameter(torch.zeros((in_features, r), device=self.weight.device, dtype=torch.bfloat16), requires_grad=True)
        self.lora_B = nn.Parameter(torch.zeros((r, out_features), device=self.weight.device, dtype=torch.bfloat16), requires_grad=True)
        nn.init.zeros_(self.lora_A)
        nn.init.normal_(self.lora_B)
        
        
    def forward(self, x):
        result = F.embedding(x, self.weight, self.padding_idx)
        after_A = F.embedding(x, self.lora_A, self.padding_idx)
        result += self.scale*(after_A@self.lora_B)
        return result



class MoeRMSNorm(nn.Module):
    def __init__(self, eps, weight_base, num_experts=1):

        super().__init__()
        self.weight = nn.ParameterList(
            [nn.Parameter(weight_base, requires_grad=True) for e in range(num_experts)]
        )
        self.variance_epsilon = eps
        self.num_experts = num_experts

    def forward(self, hidden_states, token_type):
        # hidden_statas : [B,S,H]

        results = torch.zeros_like(hidden_states)
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        # every token have their own variance in feature dim.
        variance = hidden_states.pow(2).mean(-1, keepdim=True)  # [B,S,E] -> [B,S,1]
        for expert in range(self.num_experts):        
            expert_mask = (token_type==expert)
            hidden_states[:,expert_mask,...] = hidden_states[:,expert_mask,...] * torch.rsqrt(variance[:,expert_mask,...] + self.variance_epsilon)
            results[:,expert_mask,...] = self.weight[expert] * hidden_states[:,expert_mask,...].to(input_dtype)

        return results


def replace_with_moelora_module(model, model_config):
    for name, module in model.named_children():

        if isinstance(module, nn.Linear):
            if name == "lm_head":
                setattr(model, name, LinearLoraLayer(module.in_features, module.out_features, weight=module.weight.data.clone(), r=model_config.lora_rank))
            else:
                setattr(model, name, MoeLinearLoraLayer(module.in_features, module.out_features, weight_base=module.weight.data.clone(), num_experts=model_config.num_experts, r=model_config.lora_rank))
        elif isinstance(module, nn.Embedding):
            setattr(model, name, EmbeddingLoraLayer(module.num_embeddings, module.embedding_dim, module.padding_idx, weight=module.weight.data.clone(), r=model_config.lora_rank))
        elif isinstance(module, LlamaRMSNorm):
            setattr(model, name, MoeRMSNorm(eps=module.variance_epsilon, weight_base=module.weight.data.clone(), num_experts=model_config.num_experts))
        else:
            # Recursively apply this function to submodules
            replace_with_moelora_module(module, model_config)        
        



class ReCMLLMSingleForward(torch.nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.model_config = model_config


        # modify Llama's forward function
        transformers.models.llama.modeling_llama.LlamaForCausalLM.forward = LlamaForCausalLM_forward
        transformers.models.llama.modeling_llama.LlamaModel.forward = LlamaModel_forward
        transformers.models.llama.modeling_llama.LlamaDecoderLayer.forward = LlamaDecoderLayer_forward
        transformers.models.llama.modeling_llama.LlamaSdpaAttention.forward = LlamaAttention_forward
        transformers.models.llama.modeling_llama.LlamaMLP.forward = LlamaMLP_forward
        self.model = AutoModelForCausalLM.from_pretrained(
            model_config.model_id,
            torch_dtype=torch.bfloat16,
        )

        # replace_LlamaRMSNorm_with_MoeRMSNorm(self.model)
        # replace_Linear_with_MoeLinearLoraLayer(self.model)
        # replace_Embedding_with_EmbeddingLoraLayer(self.model)
        replace_with_moelora_module(self.model, self.model_config)

        self.device = self.model.device
        print(f"init device:{self.device}")
        
        config = self.model.config
        self.vocab_size = config.vocab_size
        self.cmp_tokens = nn.Parameter(self.model.model.embed_tokens.weight.new_zeros((model_config.cmp_size, config.hidden_size)), requires_grad=True)
        self.mem_tokens = nn.Parameter(self.model.model.embed_tokens.weight.new_zeros((model_config.mem_size, config.hidden_size)), requires_grad=True)

        # self.cmp_head_num = model_config.cmp_ratio

        # self.compress_head = nn.Linear(config.hidden_size, self.cmp_head_num*config.hidden_size, bias=False, device=self.device
        #                                     dtype=self.model.model.embed_tokens.weight.dtype)

        mean = torch.mean(self.model.model.embed_tokens.weight).item()
        std = torch.std(self.model.model.embed_tokens.weight).item()
        nn.init.normal_(self.mem_tokens, mean=mean, std=std)
        nn.init.normal_(self.cmp_tokens, mean=mean, std=std)

        self.segment_begin_pid = self.model_config.groups_num*self.model_config.segment_size+1
        self.segment_end_pid = self.model_config.groups_num*self.model_config.segment_size+self.model_config.segment_size

        # pids: [1, tokens_num]
        segment_pids = get_uniform_pids(self.segment_begin_pid, self.segment_end_pid, ratio=1, device=self.device)
        cmp_pids = get_uniform_pids(self.segment_begin_pid, self.segment_end_pid, ratio=model_config.cmp_ratio, device=self.device)
        mem_pids = get_uniform_pids(self.segment_begin_pid, self.segment_end_pid, ratio=model_config.mem_ratio, device=self.device)
        mem_group_pids = torch.cat([cmp_pids, mem_pids], dim=1)

        # TOKEN TYPE: DARK_CMP, DARK_MEM; SEMANTIC; DISCARD_DARK_CMP, DISCARD_DARK_MEM; NEW_CMP, NEW_MEM
        dark_cmp = torch.full((model_config.cmp_size,), fill_value=model_config.DARK_CMP, device=self.device)
        dark_mem = torch.full((model_config.mem_size,), fill_value=model_config.DARK_MEM, device=self.device)
        semantic = torch.full((model_config.segment_size,), fill_value=model_config.SEMANTIC, device=self.device)
        discard_dark_cmp = torch.full((model_config.cmp_size,), fill_value=model_config.DISCARD_DARK_CMP, device=self.device)
        discard_dark_mem = torch.full((model_config.mem_size,), fill_value=model_config.DISCARD_DARK_MEM, device=self.device)
        new_cmp = torch.full((model_config.cmp_size,), fill_value=model_config.NEW_CMP, device=self.device)
        new_mem = torch.full((model_config.mem_size,), fill_value=model_config.NEW_MEM, device=self.device)
        dark_group = torch.cat([dark_cmp, dark_mem], dim=0)
        discard_group = torch.cat([discard_dark_cmp, discard_dark_mem], dim=0)
        new_group = torch.cat([new_cmp, new_mem], dim=0)


        # register_buffer can move the tensor to device when using model.to(device)
        self.register_buffer("segment_pids", segment_pids, persistent=False)
        self.register_buffer("cmp_pids", cmp_pids, persistent=False)
        self.register_buffer("mem_pids", mem_pids, persistent=False)
        self.register_buffer("mem_group_pids", mem_group_pids, persistent=False)

        self.register_buffer("dark_cmp", dark_cmp, persistent=False)
        self.register_buffer("dark_mem", dark_mem, persistent=False)
        self.register_buffer("semantic", semantic, persistent=False)
        self.register_buffer("discard_dark_cmp", discard_dark_cmp, persistent=False)
        self.register_buffer("discard_dark_mem", discard_dark_mem, persistent=False)
        self.register_buffer("new_cmp", new_cmp, persistent=False)
        self.register_buffer("new_mem", new_mem, persistent=False)
        self.register_buffer("dark_group", dark_group, persistent=False)
        self.register_buffer("discard_group", discard_group, persistent=False)
        self.register_buffer("new_group", new_group, persistent=False)



        


    def forward(self, inputs, past_key_values):
        
        # ->LlamaForCausalLM->LlamaModel->embed_tokens
        inputs_embeds = self.model.model.embed_tokens(inputs)
        bsz, seq_len, emb_size = inputs_embeds.size()

        # in training phase, the length of inputs is fixed. 
        assert seq_len == self.model_config.segment_size, "in training phase, the length of inputs must be fixed."

        mem_size = self.mem_tokens.size(0)
        expand_mem = self.mem_tokens.unsqueeze(0).expand(bsz, mem_size, emb_size)
        cmp_size = self.cmp_tokens.size(0)
        expand_cmp = self.cmp_tokens.unsqueeze(0).expand(bsz, cmp_size, emb_size) 

        mem_group_embeds = torch.cat([expand_cmp,expand_mem],dim=1)

        
        condition = 0
        if past_key_values.is_empty():
            whole_inputs_embeds = torch.cat([inputs_embeds, mem_group_embeds], dim=1)
            whole_pids = torch.cat([self.segment_pids, self.mem_group_pids], dim=1)
            token_type = torch.cat([self.semantic, self.new_group], dim=0)
            condition = 1
        elif past_key_values.is_full():
            whole_inputs_embeds = torch.cat([past_key_values.get_latest_embeds(), inputs_embeds, past_key_values.get_oldest_embeds(), mem_group_embeds], dim=1)
            whole_pids = torch.cat([past_key_values.get_latest_group_pids(), self.segment_pids, self.mem_group_pids, self.mem_group_pids], dim=1) 
            token_type = torch.cat([self.dark_group, self.semantic, self.discard_group,self.new_group], dim=0)
            condition = 2
        else:
            whole_inputs_embeds = torch.cat([past_key_values.get_latest_embeds(), inputs_embeds, mem_group_embeds], dim=1)
            whole_pids = torch.cat([past_key_values.get_latest_group_pids(), self.segment_pids, self.mem_group_pids], dim=1)  
            token_type = torch.cat([self.dark_group, self.semantic, self.new_group], dim=0) 
            condition = 3                   

        print(f"forward device:{inputs_embeds.device}")
        print(f"token_type:{token_type}")

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        # [B,L,E]
        hidden_state = self.model(
            position_ids=whole_pids,
            inputs_embeds=whole_inputs_embeds,
            past_key_values=past_key_values,
            token_type = token_type,
        )


        # below require that the numbers of all token_types be different
        # semantic_hidden = hidden_state[:,token_type==self.model_config["SEMANTIC"],:]
        # new_cmp_hidden = hidden_state[:,token_type==self.model_config["NEW_CMP"],:]
        # new_mem_hidden = hidden_state[:,token_type==self.model_config["NEW_MEM"],:]

        if condition == 1:
            semantic_hidden, new_group_hidden = torch.split(hidden_state,
             [self.model_config.segment_size, self.model_config.group_size], dim=1)
        elif condition == 2:
            dark_group_hidden, semantic_hidden, discard_group_hidden, new_group_hidden = torch.split(hidden_state,
             [self.model_config.group_size, self.model_config.segment_size, self.model_config.group_size, self.model_config.group_size], dim=1)
        elif condition == 3:
            dark_group_hidden, semantic_hidden, new_group_hidden = torch.split(hidden_state,
             [self.model_config.group_size, self.model_config.segment_size, self.model_config.group_size], dim=1)           
        else:
            raise ValueError("condition not in [1,2,3]")


        # [B,S,E]
        return {"semantic_hidden":semantic_hidden, "new_group_hidden":new_group_hidden}

    
    def inference_prefill(self, inputs, past_key_values):
        pass
    

    def inference_forward(self, inputs, past_key_values):
        
        # inference need cache the initial KV of current segment.
        assert past_key_value.have_current_KVcache()

# byd loss
# copy from https://github.com/huggingface/transformers/blob/main/src/transformers/loss/loss_utils.py#L24
def fixed_cross_entropy(source, target, num_items_in_batch: int = None, ignore_index: int = -100, **kwargs):
    reduction = "sum" if num_items_in_batch is not None else "mean"
    loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
    if reduction == "sum":
        loss = loss / num_items_in_batch.to(loss.device)
    return loss

## didn't record the num_items_in_batch, so use gradient_accumulation=1 or record the num_items_in_batch.


def ForCausalLMLoss(
    logits, labels, vocab_size: int, num_items_in_batch: int = None, ignore_index: int = -100, **kwargs
):
    # Upcast to float if we need to compute the loss to avoid potential precision issues
    logits = logits.float()

    # No Shift required, because have shift before.

    # Flatten the tokens
    shift_logits = logits.contiguous().view(-1, vocab_size)
    shift_labels = labels.contiguous().view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    loss = fixed_cross_entropy(shift_logits, shift_labels, num_items_in_batch, ignore_index, **kwargs)
    return loss



class ReCMLLMConfig(PretrainedConfig):
    model_type = "ReCMLLM"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class ReCMLLM(PreTrainedModel):
    config_class = ReCMLLMConfig
    def __init__(self, model_config):
        super().__init__(model_config)
        self.model_config = model_config
        self.loss_fct = ForCausalLMLoss
        self.model = ReCMLLMSingleForward(model_config)
        config = self.model.model.config
        self.compress_head = nn.Linear(config.hidden_size, model_config.cmp_ratio*config.hidden_size, bias=False, device=self.model.device,
                                            dtype=self.model.model.model.embed_tokens.weight.dtype)
        self.past_key_values = MemCache(model_config)




    def forward(self, inputs, labels, **loss_kwargs):
        # inputs, labels: [B,S]
        bsz, seq_len = inputs.size()
        seg_len = self.model_config.segment_size
        segment_num = seq_len//seg_len
        assert segment_num*seg_len == seq_len, f"in training phase, The input seq_len is an integer multiple of the segment size, {segment_num},{seg_len},{seq_len}"

        whole_hidden = []
        whole_cmp_hidden = []
        
        
        for i in range(segment_num):
            print("--"*50)
            print(f"seg_id:{i}/{segment_num}")
            outputs = self.model(inputs[:, i*seg_len:(i+1)*seg_len], self.past_key_values)
            whole_hidden.append(outputs["semantic_hidden"])
            cmp_hidden = self.compress_head(outputs["new_group_hidden"][:,:self.model_config.cmp_size])
            # [B,mem_size,head_num*hidden_dim] -> [B,mem_size*head_num, hidden_dim]
            # print(f"cmp_hidden:{cmp_hidden.shape};cmp_hidden.view:{cmp_hidden.view(bsz, self.model_config.cmp_size*self.model_config.cmp_ratio, -1).shape}")
            whole_cmp_hidden.append(cmp_hidden.view(bsz, self.model_config.cmp_size*self.model_config.cmp_ratio, -1))
            self.past_key_values.update_group_hidden(outputs["new_group_hidden"], self.model.mem_group_pids)

        # list of [B,Seg_len,E]
        print(f"torch.cat(whole_cmp_hidden, dim=1):{torch.cat(whole_cmp_hidden, dim=1).shape}")
        whole_logits = self.model.model.lm_head(torch.cat(whole_hidden, dim=1))
        whole_cmp_logits = self.model.model.lm_head(torch.cat(whole_cmp_hidden, dim=1))

        #   inputs  : 1,2,  3 ,4,5,  6 ,  7 ,8,  9  ,<eos>,<eos>,<eos>
        #   labels  : 2,3,-100,5,6,-100,-100,9,<eos>,-100 ,-100 ,-100
        # cmp_labels: 1,2,  3 ,4,5,  6 ,  7 ,8,  9  ,<eos>,-100 ,-100   this choice:only compress valid info
        # cmp_labels: 1,2,  3 ,4,5,  6 ,  7 ,8,  9  ,<eos>,<eos>,<eos>  this choice:compress what compression token have seen. (*) I decise usint this choice.
        cmp_labels = inputs.detach().clone()

        # warnings.warn("num_items_in_batch is not provided at this time, please confirm gradient_accummulation equal to 1")
        print(f"loss_kwargs:{loss_kwargs}")

        # causal_lm_loss
        causal_lm_loss = self.loss_fct(logits=whole_logits, labels=labels, vocab_size=self.model.model.config.vocab_size, **loss_kwargs)
        
        # cmp_loss
        cmp_loss = self.loss_fct(logits=whole_cmp_logits, labels=cmp_labels, vocab_size=self.model.model.config.vocab_size, **loss_kwargs)

        return {
            "loss":self.model_config.causal_lm_weight*causal_lm_loss + self.model_config.cmp_weight*cmp_loss,
            "lm_logits":whole_logits,
            "cmp_logits":whole_cmp_logits,
            "causal_lm_loss":causal_lm_loss.item(),
            "cmp_loss":cmp_loss.item()
        }


# def save_adapter(model,save_path_and_name='adapter.pt', log=False):
#     adapter_name = set()
#     for name, param in model.named_parameters():
#         if param.requires_grad:
#             if log:
#                 print("[Save Adapter]",name)
#             adapter_name.add(name)
            
#     state_dict = model.state_dict()
#     adapter_state_dict = {name: param for name, param in state_dict.items() if name in adapter_name}
#     torch.save(adapter_state_dict, save_path_and_name)

# def load_adapter(model, save_path_and_name='adapter.pt', log=False):
#     adapter_state_dict = torch.load(save_path_and_name, map_location='cpu')  # 先加载到CPU
#     if log:
#         print("Loading adapter parameters:")
#         for name, _ in adapter_state_dict.items():
#             print(f"[Load Adapter] {name}")
    
#     # 将adapter的权重转移到模型的设备上
#     adapter_state_dict = {k: v.to(model.device) for k, v in adapter_state_dict.items()}
    
#     model.load_state_dict(adapter_state_dict, strict=False)
#     return model




# def load_model_with_adapter(model_id, task_config, rank, save_path_and_name='adapter.pt', log=False):
#     model = get_model(model_id, task_config, rank)
#     load_adapter(model, save_path_and_name, log)
#     return model

def load_json(path):
    with open(path, 'r') as f:
        obj = json.load(f)
    return obj


if __name__ == "__main__":
    # mask = xops.fmha.attn_bias.LowerTriangularFromBottomRightMask().materialize(shape=(2,4,8))
    # print(mask)

    AutoConfig.register("ReCMLLM", ReCMLLMConfig)
    AutoModel.register(ReCMLLMConfig, ReCMLLM)

    model_config = load_json("./ReCMLLM/config.json")
    model_config = ReCMLLMConfig(**model_config)
    print(model_config)
    model = ReCMLLM(model_config)
    print(model)
    # model.to('cuda')
    # print(model.device)

    data = torch.load("mini_data.pt", weights_only=False)
    print(data[0])
    model.train()
    model.to("cuda")
    output = model(inputs=data[0]["input_ids"].unsqueeze(0).to("cuda"), labels=data[0]["labels"].unsqueeze(0).to("cuda"))

    print(output)

    # model = ReCMLLM.from_pretrained(model_name_or_path, config=config)
    # print("-"*100)
    # print(model)
    


"""
python modeling.py
"""
    