# from transformers.cache_utils import Cache
import torch
import os
class MemCache(torch.nn.Module):

    def __init__(self, model_config):
        super().__init__()
        self.config = model_config

        self.group_len = model_config.cmp_size+model_config.mem_size
        self.max_len = self.group_len*model_config.groups_num
        self.groups_pids = None  # [1, seq_len]
        self.groups_hidden_cache = None  # [batch_size, seq_len, hidden_dim]
        self.groups_key_cache = []   # List(layer_dims) of [batch_size, seq_len, kv_head_num, head_dim],  
        self.groups_value_cache = []

        self.segment_len = model_config.segment_size
        self.segment_pids = None
        self.segment_key_cache = []
        self.segment_value_cache = []

    def is_empty(self):
        return self.groups_hidden_cache is None    

    def is_full(self):
        return self.groups_hidden_cache is not None and self.groups_hidden_cache.size(1)==self.max_len

    def kv_cache_is_empty(self):
        return len(self.groups_key_cache) == 0

    def get_latest_embeds(self):
        # [batch_size, seq_len, hidden_dim]
        return self.groups_hidden_cache[:,-self.group_len:,:]

    def get_oldest_embeds(self):
        # [batch_size, seq_len, hidden_dim]
        return self.groups_hidden_cache[:,:self.group_len,:]

    def get_latest_group_pids(self):
        # [1, seq_len]
        return self.groups_pids[:,-self.group_len:]

    def get_kvcache_pids(self):
        return self.groups_pids[:,:-self.group_len]

    def update_group_hidden(self, group_hidden, group_pids):
        # [batch_size, seq_len, hidden_dim]
        if self.is_empty():
            self.groups_hidden_cache = group_hidden
            self.groups_pids = group_pids
        elif self.is_full():
            self.groups_hidden_cache = torch.cat([self.groups_hidden_cache[:,self.group_len:,:], group_hidden],dim=1)
            self.groups_pids = torch.cat([self.groups_pids[:,self.group_len:], group_pids],dim=1)
            for i in range(len(self.groups_key_cache)):
                self.groups_key_cache[i] = self.groups_key_cache[i][:,self.group_len:,...]
                self.groups_value_cache[i] = self.groups_value_cache[i][:,self.group_len:,...]
            
        else:
            self.groups_hidden_cache = torch.cat([self.groups_hidden_cache, group_hidden],dim=1)
            self.groups_pids = torch.cat([self.groups_pids, group_pids],dim=1)

        # shift for next segment
        self.groups_pids -= self.segment_len


    def update(
        self,
        key_states,
        value_states,
        layer_idx,
        cache_kwargs = None,
    ):
        """
        Input:
        key_states, value_states:[batch_size, seq_len, kv_head_num, head_dim]

        return : 
        key_states, value_states:[batch_size, cache_len+seq_len, kv_head_num, head_dim]
        """
    
        # Update the cache

        if self.is_empty():  # inputs_embs : [inputs_embeds, mem_group_embeds], nothing require caching.
            return key_states, value_states
        #  inputs_embs : [latest_embeds, inputs_embeds, oldest_embeds, mem_group_embeds] or [latest_embeds, inputs_embeds, mem_group_embeds]
        elif len(self.groups_key_cache) <= layer_idx:
            self.groups_key_cache.append(key_states[:,:self.group_len,...])
            self.groups_value_cache.append(value_states[:,:self.group_len,...])
            return key_states, value_states
        else:

            return_key_states = torch.cat([self.groups_key_cache[layer_idx], key_states], dim=1)
            return_value_states = torch.cat([self.groups_value_cache[layer_idx], value_states], dim=1)
            self.groups_key_cache[layer_idx] = torch.cat([self.groups_key_cache[layer_idx], key_states[:,:self.group_len,...]], dim=1)
            self.groups_value_cache[layer_idx] = torch.cat([self.groups_value_cache[layer_idx], value_states[:,:self.group_len,...]], dim=1)

            return return_key_states, return_value_states
