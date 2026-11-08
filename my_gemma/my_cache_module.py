import torch
import torch.nn as nn

class my_Cache:
    """
    A `Cache` is mostly a list of `CacheLayerMixin` objects, one per model layer. It serves as a container for
    the Cache of each layer.

    Args:
        layers (`Optional`, *optional*):
            A list of pre-created `CacheLayerMixin` or `LinearAttentionCacheLayerMixin`. If omitted (`None`), then `layer_class_to_replicate`
            will be used.
    """

    def __init__(self):
        self.layers:list[list[torch.Tensor, torch.Tensor]] =[] #list[list[torch.Tensor, torch.Tensor]]
        self.cache_position=[]#(batch,num_posi1)
        self.causal_mask=[]#[b,1, positions]
        self.is_first_iteration=True

    def set_cache_position(self, prefilling_position_ids):
        """
        slice the prefilling_position_ids(batch,num_posi1?) to -> (batch,num_posi=1)
            prefilling_position_ids:(batch,num_posi1?)
        Return:
            cache_position:(batch,num_posi=1)
        """
        #(batch,num_posi1?)->(batch,))->(batch,1)
        self.cache_position=(prefilling_position_ids[:,-1])[:,None]
    def get_cache_position(self):
        self.cache_position+=1
        return self.cache_position


    def set_cache_mask(self, causal_mask):
        """
        slice the prefilling_position_ids(batch,num_posi1?) to -> (batch,num_posi=1)
            causal_mask:(b,heads=1,num_query,num_key)
        Return:
            causal_mask:(b,heads=1,num_query=1,num_key)
                    broastable with attn_weights(b,att_heads,num_posi_query?1,num_posi_key)
        """
        #(b,heads=1,num_query,num_key)->(b,heads=1,num_key)->(b,heads=1,num_query=1,num_key)
        self.causal_mask=(causal_mask[:,:,-1,:])[:,:,None,:]


    def update_cache_gen_mask(self):
        b=self.causal_mask.shape[0]
        appending=torch.ones([b, 1,1,1], dtype=torch.int16)
        self.causal_mask=torch.cat((self.causal_mask, appending), dim=-1)
        return self.causal_mask


    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int
    ) -> list[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.
        Parameters:
            key_states:(batch,num_kv_heads,num_posi_1?,head_dim)
            value_states:(batch,num_kv_heads,num_posi_1?,head_dim)
            layer_idx:int
        Return:
            A list :(batch,num_kv_heads,num_posi,head_dim)
        """
        if len(self.layers) <= layer_idx:#if it's the prefilling step, each layer just appending (keys, values)
            idx_pair=[key_states, value_states]
            self.layers.append(idx_pair)
        else:#the auto regressive generation step
            old_keys=self.layers[layer_idx][0]
            self.layers[layer_idx][0]=torch.cat((old_keys, key_states), 2)#(batch,kv_heads,num_posi_1?,head_dim)->(batch,kv_heads,num_posi,head_dim)
            old_values=self.layers[layer_idx][1]
            self.layers[layer_idx][1]=torch.cat((old_values, value_states), 2)

        keys=self.layers[layer_idx][0]
        values=self.layers[layer_idx][1]

        return keys, values


    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Returns the sequence length of the cache for the given layer."""
        if layer_idx >= len(self.layers):
            return 0

        seq_length=self.layers[layer_idx][0].shape[2]#(batch,kv_heads,num_posi,head_dim)

        return seq_length
