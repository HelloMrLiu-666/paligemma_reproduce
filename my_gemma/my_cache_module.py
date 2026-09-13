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
        self.layers:list[tuple[torch.Tensor, torch.Tensor]] =[] #list[tuple[torch.Tensor, torch.Tensor]]

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.
        Parameters:
            key_states:(batch,num_kv_heads,num_posi_1?,head_dim)
            value_states:(batch,num_kv_heads,num_posi_1?,head_dim)
            layer_idx:int
        Return:
            A tuple :(batch,num_kv_heads,num_posi,head_dim)
        """
        if len(self.layers) <= layer_idx:#if it's the prefilling step, each layer just appending (keys, values)
            idx_tuple=(key_states, value_states)
            self.layers.append(idx_tuple)
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
