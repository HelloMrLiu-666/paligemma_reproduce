from typing import Callable
import torch
import torch.nn as nn
from my_configuration_gemma import my_GemmaConfig
from my_cache_module import my_Cache
from transformers import AutoTokenizer, GemmaForCausalLM


#for benchmarking debugging 
# import transformers.models.gemma.modeling_gemma as modeling_gemma
# import transformers.generation.logits_process as logits_process
import transformers.generation.utils as utils
#this is for liu_guang debugging
debug_values = {}
debug_values["gen_step"] =0
debug_values["deco_layer"] =0


class GemmaTextScaledWordEmbedding(nn.Embedding):
    """
    This module overrides nn.Embeddings' forward by multiplying with embeddings scale.
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int, embed_scale: float = 1.0):
        super().__init__(num_embeddings, embedding_dim, padding_idx=padding_idx)
        self.scalar_embed_scale = embed_scale
        self.register_buffer('embed_scale', torch.tensor(embed_scale, dtype=self.weight.dtype), persistent=False)

    def forward(self, input_ids: torch.Tensor)->torch.Tensor:
        """
        Args:
            input_ids:[batch, num_posi]
        Returns:
            input_emb:[batch, num_posi,emb_dim]
        """
        input_emb=self.embed_scale*(super().forward(input_ids))

        return input_emb


class GemmaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(data=torch.ones((dim),  dtype=torch.float32))

    def _norm(self, x):
        """
        Args:
            x:[batch, num_posi,emb_dim]
        Returns:
            normalized:[batch, num_posi,1]
        """
        x=x.pow(2).mean(dim=-1, keepdim=True)#[batch, num_posi,emb_dim]->[batch, num_posi,1]
        normal_factor=torch.rsqrt(x+self.eps)#no shape change
        return normal_factor

    def forward(self, x):
        # with torch.autocast(device_type=x.device_type, enabled=False):
        output = x*self._norm(x.float())#[batch, num_posi,emb_dim]*[batch, num_posi,1]->[batch, num_posi,emb_dim]
        output = output*(1+self.weight)#[batch, num_posi,emb_dim]*[1, 1,emb_dim]->[batch, num_posi,emb_dim]
        return output.type_as(x)

class GemmaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn =nn.GELU(approximate='tanh')

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x))*self.up_proj(x))
        return down_proj


class GemmaRotaryEmbedding(nn.Module):
    def __init__(self, config: my_GemmaConfig):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        # self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        rope_init_fn: Callable = self.compute_default_rope_parameters
        inv_freq = rope_init_fn(self.config)

        self.inv_freq = nn.Buffer(inv_freq, persistent=False)
        # self.original_inv_freq = 

    @staticmethod
    def compute_default_rope_parameters(config: my_GemmaConfig, **kwargs) -> torch.Tensor.float:
        """
        Computes the inverse frequencies according to the original RoPE implementation
        Args:
            config
        Returns:
            torch.Tensor: (head_dim/2,)  containing the inverse frequencies for the RoPE embeddings 
        """
        dim=config.hidden_size//config.num_attention_heads

        # #to device is not set for the tensor here, so the device need to be set within forward before operation with GPU tensor x
        # #otherwise the device should be desiged as a separate argument to pass through multiple functions to get to here
        exponents=torch.arange(0,dim,2, dtype=torch.float)#even though looks like not the same as paper, the final range of the exponents is the same as the paper [0,dim-2]
        theta=1.0/(10000.0**(exponents/float(dim)))
        return theta


    def forward(self, x, position_ids):
        """
        Args:
            x: [batch,heads, num_posi,head_dim]
            position_ids:(b,num_posi_1?)

        Returns:
            cos : [b,len=1?,head_dim]
            sin : [b,len=1?,head_dim]
        """
        #my implementation is different to huggingface, I haven't include batch information in cos & sin
        position_ids_expanded=position_ids[:,None,:].float()#(b,num_posi_1?)->(b,1,num_posi_1?)
        #(head_dim/2,)->(1,head_dim/2,1)->(b,head_dim/2,1)
        theta=(self.inv_freq)[None,:,None].expand(position_ids_expanded.shape[0],-1,1).float().to(device=x.device)

        with torch.autocast(device_type=x.device.type, enabled=False):
            #(b,head_dim/2,1)@(b,1,num_posi_1?)->(b,head_dim/2,num_posi_1?)->(b,num_posi_1?,head_dim/2)
            m_theta=(theta@position_ids_expanded).transpose(1,2)
            cos=torch.cos(m_theta)
            sin=torch.sin(m_theta)
            cos=torch.cat((cos, cos), dim=-1)#(b,len=1?,head_dim/2)->(b,len=1?,head_dim)
            sin=torch.cat((sin, sin), dim=-1)

        return cos.type_as(x),sin.type_as(x)

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    """
    Args:
        x: [batch,heads, num_posi,head_dim]
    Returns:
        x: [batch,heads, num_posi,head_dim]
    """
    _,_,_,head_dim=x.shape
    x1=x[...,:head_dim//2]
    x2=x[...,head_dim//2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): (batch,heads, len=1?,head_dim)
        k (`torch.Tensor`): (batch,heads, len=1?,head_dim)
        cos (`torch.Tensor`): (b,len=1?,head_dim)
        sin (`torch.Tensor`): (b,len=1?,head_dim)
        unsqueeze_dim (`int`, *optional*, defaults to 1):
    Returns:
        q_embed: (batch,heads, len=1?,head_dim)
        k_embed: (batch,heads, len=1?,head_dim)
    """
    q_rotate_half=rotate_half(q)
    k_rotate_half=rotate_half(k)    #(batch,heads, len=1?,head_dim)
    cos=cos[:,None,:,:]#(b,len=1?,head_dim)->(b,1,len=1?,head_dim)
    sin=sin[:,None,:,:]
    q_embed=(q*cos)+(rotate_half(q)*sin)
    k_embed=(k*cos)+(rotate_half(k)*sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    Args:
        hidden_states: (batch,kv_heads,num_posi,head_dim)
    Returns:
        hidden_states: (batch,att_heads,num_posi,head_dim)
    """
    batch,kv_heads,num_posi,head_dim=hidden_states.shape
    hidden_states=hidden_states[:,:,None,:,:].expand(-1,-1,n_rep,-1,-1).contiguous()
    return hidden_states.view(batch,-1,num_posi,head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
):
    """
    Args:
        query: (batch,att_heads,num_posi_query,head_dim)
        key: (batch,kv_heads,num_posi_key,head_dim)
        value: (batch,kv_heads,num_posi,head_dim)
        attention_mask: broastable with attn_weights(batch,att_heads,num_posi_query,num_posi_key)
                        unlike siglip whose attention_mask defined in this method, attention_mask is defined @ training/inference runtime
    Returns:
        attn_output: (batch,att_heads,num_posi_query,head_dim)
        attn_weights:(batch,att_heads,num_posi_query,num_posi_key)
    """
    n_rep=module.num_key_value_groups
    key_states=repeat_kv(key,n_rep )
    value_states=repeat_kv(value, n_rep)

    #to align with huggingface implementation, convert the bool type mask to tensor with 0 and -inf
    min_dtype = torch.finfo(torch.float32).min
    mask =torch.zeros_like(attention_mask, dtype=torch.float32)
    mask = mask.masked_fill(~attention_mask, min_dtype)

    #(batch,att_heads,num_posi_query,head_dim)@(batch,att_heads,head_dim,num_posi_key)->(batch,att_heads,num_posi_query,num_posi_key)
    attn_weights=torch.matmul(query, key_states.transpose(2, 3))*scaling
    if not attention_mask is None:
        attn_weights=attn_weights+mask

    with torch.autocast(device_type=query.device.type, enabled=False):
        attn_weights=nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
    attn_weights=attn_weights.to(query.dtype)#only the softmax is calculated as float32, other matrix multiplication might in lower precision
    attn_weights=nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    #(batch,att_heads,num_posi_query,num_posi_key)@(batch,att_heads,num_posi_key,head_dim)->(batch,att_heads,num_posi_query,head_dim)
    #(batch,att_heads,num_posi_query,head_dim)->(batch,num_posi_query,att_heads,head_dim)
    attn_output=torch.matmul(attn_weights, value_states).transpose(1,2)
    return attn_output, attn_weights


class GemmaAttention(nn.Module):
    def __init__(self, config: my_GemmaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = not getattr(config, "use_bidirectional_attention", False)

        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads*self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads*self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads*self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(config.num_attention_heads*self.head_dim, config.hidden_size, bias=config.attention_bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: my_Cache | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: (batch,num_posi_1?,emb_dim)
            position_embeddings: cos, sin of shape [num_posi_1?,head_dim]
            attention_mask: broastable with attn_weights(batch,att_heads,num_posi_query,num_posi_key)
            past_key_values: 

        Returns:
            attn_output: (batch,num_posi_1?,hidden_size) ,num_posi_query is num_posi_1?
            attn_weights:(batch,att_heads,num_posi_query,num_posi_key)
        """

        q=self.q_proj(hidden_states)#(batch,num_posi_1?,emb_dim)->(batch,num_posi_1?,num_att_heads*head_dim)
        k=self.k_proj(hidden_states)#(batch,num_posi_1?,emb_dim)->(batch,num_posi_1?,num_kv_heads*head_dim)
        v=self.v_proj(hidden_states)#(batch,num_posi_1?,emb_dim)->(batch,num_posi_1?,num_kv_heads*head_dim)

    #reshape q & k & v
        shape=hidden_states.shape[:-1]#(batch,num_posi_1?)
        hidden_shape=(*shape,-1,self.head_dim)
        ##(batch,num_posi_1?)->(batch,num_posi_1?,num_att_heads,head_dim)->(batch,num_att_heads,num_posi_1?,head_dim)
        q_states=q.view(hidden_shape).transpose(1,2)
        k_states=k.view(hidden_shape).transpose(1,2)#(batch,num_kv_heads,num_posi_1?,head_dim)
        v_states=v.view(hidden_shape).transpose(1,2)#(batch,num_kv_heads,num_posi_1?,head_dim)

    #apply the position embedding before the store/get info into/from kv cache
        cos, sin= position_embeddings#[num_posi_1?,head_dim]
        q_states,k_states=apply_rotary_pos_emb(q_states, k_states, cos, sin)#no shape change for q k

    #store/get info into/from kv cache
        #(batch,num_kv_heads,num_posi_1?,head_dim)->(batch,num_kv_heads,num_posi,head_dim)
        k_states, v_states=past_key_values.update(k_states, v_states, self.layer_idx)

    #align the heads
        attn_output, attn_weights=eager_attention_forward(#no shape change for q_state (batch,num_att_heads,num_posi_1?,head_dim)
            module=self,
            query=q_states,
            key=k_states,
            value=v_states,
            attention_mask=attention_mask,
            scaling=self.scaling,
            dropout=self.attention_dropout)

        #(batch,num_posi_1?,num_att_heads,head_dim)->(batch,num_posi_1?,num_att_heads*head_dim)
        attn_output=attn_output.contiguous().view(*shape,-1)
        #(batch,num_posi_1?,num_att_heads*head_dim)->(batch,num_posi_1?,hidden_size)
        attn_output=self.o_proj(attn_output)

        return attn_output, attn_weights


class GemmaDecoderLayer(nn.Module):
    def __init__(self, config: my_GemmaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = GemmaAttention(config, layer_idx)

        self.mlp = GemmaMLP(config)
        self.input_layernorm = GemmaRMSNorm(self.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(self.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        # position_ids: torch.LongTensor | None = None,
        past_key_values: my_Cache | None = None,
        # use_cache: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None, #the position embedding is calculated by rotary_emb layer in GemmaModel
    ) -> torch.Tensor:

        residual=hidden_states
        hidden_states=self.input_layernorm(hidden_states)

        hidden_states,_=self.self_attn(hidden_states,position_embeddings,attention_mask,past_key_values)

        hidden_states=hidden_states+residual

        residual=hidden_states
        hidden_states=self.post_attention_layernorm(hidden_states)
        hidden_states=self.mlp(hidden_states)
        hidden_states=hidden_states+residual
        return hidden_states



class my_GemmaModel(nn.Module):
    def __init__(self, config: my_GemmaConfig):
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        # Gemma3 downcasts the below to bfloat16, causing sqrt(3072)=55.4256 to become 55.5. See https://github.com/huggingface/transformers/pull/29402
        self.embed_tokens = GemmaTextScaledWordEmbedding(
            self.vocab_size,config.hidden_size,self.padding_idx, config.hidden_size**0.5)
        self.layers = nn.ModuleList(
            [GemmaDecoderLayer(config=config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self.norm = GemmaRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = GemmaRotaryEmbedding(config)

    def get_input_embeddings(self):#to be accessed by other models(paligemma)
        return self.embed_tokens

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor, 
        #past_key_values are defined as argument within GemmaModel&GemmaForCausalLM
        #cause during autoregressive inferencing, this forward would be called many times, so these calling should share global past_key_values
        #also, for attention_mask, cause gemma would be used within paligemma, who have special attention_mask specific to paligemma
        #but for position_ids, it's dependent on input_ids&past_key_values, could generated within this forward function
        #in this implementation, in order to align with huggingface implementation, position_ids still designed as argument
        position_ids: torch.LongTensor,
        past_key_values: my_Cache ,
        inputs_embeds: torch.FloatTensor | None = None,#gemma model could be used by paligemma,which produce it own embedding with image info
    ) ->tuple[torch.Tensor, my_Cache]:
        """
        Args:
            input_ids: (batch,num_posi_1?)
            position_ids: (batch,num_posi_1?)
            attention_mask: broastable with attn_weights(batch,att_heads,num_posi_query,num_posi_key)
            inputs_embeds: (batch, num_posi_1?,emb_dim)
            past_key_values: Cache class
        Returns:
            last_hidden_state: (batch, num_posi_1?,emb_dim)
            past_key_values: Cache class
        """
        if inputs_embeds==None:
            embeds=self.embed_tokens(input_ids)#(batch,num_posi_1?)->(batch,num_posi_1?,emb_dim)
        else:#gemma model could be used by paligemma,which produce it own embedding with image info
            embeds=inputs_embeds

        #use the position_id to generate position_embeddings->(cos, sin)
        position_embeddings=self.rotary_emb(embeds, position_ids)
        hidden_states=embeds



        # cast it to Boolean tensor
        attention_mask=attention_mask.type('torch.BoolTensor')

# #for benchmarking debugging, for multiple inference/generation steps, each step have different mask
#         debug_values["gen_step"]+=1     
#         #for benchmarking debugging 
#         if (debug_values["gen_step"]-1)==modeling_gemma.debug_values["gen_step_to_check"]:
#             # debug_values["debugging"] =(attention_mask.detach().clone()) #hidden_dim.detach().clone() 
#             if (debug_values["gen_step"]-1)>0:
#                 #(batch,num_kv_heads,num_posi_1?,head_dim)->(new_dimension, batch,num_kv_heads,num_posi_1?,head_dim)
#                 constructured_keys=torch.stack([layer_i[0] for layer_i in past_key_values.layers], dim=0)
#                 constructured_values=torch.stack([layer_i[1] for layer_i in past_key_values.layers], dim=0)
#                 debug_values["debugging"] =(constructured_keys.detach().clone()) #hidden_dim.detach().clone() 

        for layer in self.layers:
            hidden_states=layer(
                hidden_states,
                attention_mask,
                past_key_values,
                position_embeddings
            )

        hidden_states=self.norm(hidden_states)#no shape change\
        last_hidden_state=hidden_states

        return last_hidden_state



class my_GemmaForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config=config
        self.model = my_GemmaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, self.vocab_size, bias=False)
    def get_input_embeddings(self):#to be accessed by other models(paligemma)
        return self.model.get_input_embeddings()
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        position_ids: torch.LongTensor,
        past_key_values: my_Cache,
        # inputs_embeds: torch.FloatTensor | None = None,
        # labels: torch.LongTensor | None = None,
        # logits_to_keep: int | torch.Tensor = 0,
        # **kwargs: Unpack[TransformersKwargs],
    ) ->torch.Tensor:
        """
        Args:
            input_ids: (batch,num_posi_1?)
            attention_mask: broastable with attn_weights(batch,att_heads,num_posi_query,num_posi_key)
            past_key_values: Cache class
        Returns:
            logits: (batch, num_posi_1?,vocab)
        """
        hidden_states=self.model(input_ids,attention_mask,position_ids,past_key_values)#[0]

        #align with huggingface implementation, only compute the logits of last position
        logits=self.lm_head(hidden_states[:,-1:,:])#(batch, num_posi=1,emb_dim)->(batch, num_posi=1,vocab)

        return logits

    #essentiallly, huggingface implementation, the generation logic inherits from GenerationMxin class into PaliGemmaForConditionalGeneration
    #so, in my implementation, the generation logic also put within generate function and within PaliGemmaForConditionalGeneration class,
    #to mimic the usage of this the generation model as huggingface implementation, design this generate function
    #input_ids/attention_mask/position_ids are prepared in this function and passed to the forward function
    def generate( #the input of this function is the output of PaligemmaProcessor
        self,
        input_ids: torch.LongTensor, 
        attention_mask: torch.Tensor = None, #is the padding mask from the tokenizer
        # pixel_values: torch.FloatTensor, #specific to paligemma input, no need for gemma
        generation_len: int | None = None,
        top_p: float=0.9):

        generated_ids=None
        pad_token_id=self.config.pad_token_id
        #the stoping criteria currently only based on generation_len, not considering the eos token reached 
        eos_token_id=self.config.eos_token_id
        batch_eos_reached=False

        #initilaize the past_key_values
        past_key_values=my_Cache()
        #prefilling & auto-regressive generation
        for i in torch.arange(generation_len):# starting from 0
            #setting if it's the prefilling step, and initialize output_ids
            if i ==0:
                is_first_iteration=True
                past_key_values.is_first_iteration=is_first_iteration
                #initialize a padding tensor of shape (b,generation_len)
                generated_ids=torch.full((input_ids.shape[0],generation_len), pad_token_id, dtype=input_ids.dtype,device=input_ids.device)
            else:
                is_first_iteration=False
                past_key_values.is_first_iteration=is_first_iteration

            #each generation step, updates the attention_mask,position_ids 
            attention_mask,position_ids=prepare_inputs_for_generation(#the returned mask is the paligemma specific mask
                input_ids,
                past_key_values,
                attention_mask,#is the padding mask from the tokenizer
                is_first_iteration)

            # logits=self.forward(  #this is for paligemma specific generation function
            #     input_ids, 
            #     attention_mask,
            #     position_ids,
            #     past_key_values,
            #     pixel_values)

            logits=self.forward( #this is for gemma specific generation function 
                input_ids,
                attention_mask,
                position_ids,
                past_key_values)
            




        #from logits to id in vocab, and update to the input_ids
            #(batch, num_posi=1,vocab_size)->(batch, num_posi=1,1)
            next_token_id=self.my_sample_top_p(logits,top_p)
            #update the input_ids to prepare it for the auto-regressive generation
            #[:,-1,:] is to slice last predicted token, for the first step prefilling it's obvious
            #but the auto regressive generation step also needed because the output
            #shape would be (batch, 1,1), still need the slicing to make it 2 dimensional
            input_ids=next_token_id[:,-1,:] 
            #fill the next_token_id to generated_ids ,(b,generation_len), (batch, num_posi1?)
            generated_ids[:,i]=next_token_id[:,-1,:] 

        return generated_ids

            
    def my_sample_top_p(self,logits, p):# (b,seq,vocab)->(b,seq,1)
        b,s,v=logits.shape

        #sort, (b,seq,vocab), no shape change
        sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=False)
        #softmax and cumsum, no shape change
        cumulative_probs =nn.functional.softmax(sorted_logits, dim=-1, dtype=torch.float32).cumsum(dim=-1)
        # bool mask to the sorted posiitons (True means to -inf i.e. remove)
        sorted_indices_to_remove =(cumulative_probs<=(1-p))
        sorted_indices_to_remove[:,:,-1:]=False # Keep at least 1 lefted indice

        # transfer the sorted_indices_to_remove, to the unsorted/original bool mask(indices_to_remove);
        # the connection of them are the sorted_indices
        # whose matrix index position is corresponding to the index position of sorted_indices_to_remove
        # whose value corresponding to the indices_to_remove's index position along vocabulary dimension
        true_tensor = torch.ones(sorted_logits.shape, dtype=torch.bool)
        indices_to_remove =true_tensor.scatter(-1, sorted_indices, sorted_indices_to_remove)

        #mask_fill the remove position to be -inf
        scores_processed =logits.masked_fill_(indices_to_remove, value=-float("inf"))

    #for benchmarking debugging
        g_cpu = torch.Generator(device='cpu')
        g_cpu.manual_seed(42)

        probs =nn.functional.softmax(scores_processed, dim=-1, dtype=torch.float32)
        #sample from the top P, recent version pytorch support 3D input tensor for multinomial, the last dim is the sampling dim
        probs=probs.view(-1,v)
        #after debugging, need to remove  the generator
        probs_index=torch.multinomial(probs, num_samples=1,generator=g_cpu)#(b*seq,vocab)->index (b*seq,1) 
        next_token=probs_index.view(b,s,1)#the multinomial is sandwiched because multinomial support upto 2D

    # #for benchmarking debugging, for multiple inference/generation steps, each step have different mask
    #     debug_values["gen_step"]+=1     
    #     #for benchmarking debugging 
    #     if (debug_values["gen_step"]-1)==utils.debug_values["gen_step_to_check"]:
    #         debug_values["debugging"] =(probs.detach().clone()) #hidden_dim.detach().clone() 

        return next_token

def prepare_inputs_for_generation(
            input_ids: torch.LongTensor,
            # next_sequence_length: int | None = None,
            past_key_values: my_Cache | None = None,
            attention_mask: torch.LongTensor | None = None, #is the padding mask from the tokenizer
            is_first_iteration: bool | None = False,
        ):
    """
    
    Prepare the model inputs for generation. Notable steps include selecting the correct input key and cloning when appropriate,
    creating position_ids from the attention_mask when missing, slicing inputs and converting 2D attention masks to 4D for
    compilable caches, and finally forwarding all additional keyword arguments unchanged to the model's forward pass.

    See the forward pass in the model documentation for expected arguments (different models might have different
    requirements for e.g. `past_key_values`). This function should work as is for most LLMs.
    
    Args:
        input_ids:(batch, num_posi1?)
    Returns:
        attention_mask:(batch,att_heads,num_posi_query?1,num_posi_key)
        position_ids:(b, num_posi1?)
    """
    batch,q_len=input_ids.shape
    # position_ids, following is to mimic the huggingface model position_id generation
    if is_first_iteration:
        position_ids = attention_mask.long().cumsum(-1) - 1#same shape as the attention_mask:(batch,num_posi1?)
        # We need this as otherwise padding tokens appear as -1 in position
        position_ids = position_ids.masked_fill(attention_mask == 0, 0)#(batch,num_posi>1)
        #(batch,num_posi1?)->(batch,1)
        past_key_values.set_cache_position(position_ids)
    else:
        position_ids=past_key_values.get_cache_position()#(batch,num_posi=1)
    position_ids=position_ids#+1#in Paligemma are 1-indexed


    # Create the mask, is to mimic the create_causal_mask function of huggingface GemmaModel class
    if is_first_iteration:#the shape should be 
        b,positions=attention_mask.shape
        whole_matrix=torch.zeros([b,positions, positions], dtype=torch.int16)#whole faulse matrix
        no_pad_count=torch.sum(attention_mask, dim=-1, keepdim=False)#(batch,num_posi1?)->(batch,)
        for count in no_pad_count:
            i=0
            corner_matrix=torch.ones([count, count], dtype=torch.int16)
            corner_matrix=torch.tril(corner_matrix, diagonal=0)#
            whole_matrix[i,positions-count:,positions-count:]=corner_matrix[:,:]
            i+=1
        causal_mask=whole_matrix
        #[b,positions_q, positions_v]->(b,heads=1,num_query,num_key)
        causal_mask=whole_matrix[:,None,:,:]
        #(b,heads=1,num_query,num_key)->(b,heads=1,num_query=1,num_key)
        past_key_values.set_cache_mask(causal_mask)
    else:
        causal_mask=past_key_values.update_cache_gen_mask()





    # # Create the mask, only works for no padding existed in the input_ids
    # if is_first_iteration:
    #     attention_mask=torch.full((batch,q_len,q_len), 0.0, dtype=torch.int16,device=input_ids.device)
    # else:
    #     assert q_len == 1
    #     kv_len =past_key_values.get_seq_length(layer_idx=0) + q_len
    #     attention_mask=torch.full((batch,q_len,kv_len), 0.0, dtype=torch.int16,device=input_ids.device)

    return causal_mask,position_ids



# if __name__ == "__main__":

# #huggingface GemmaModel initialization
#     if debug_values["gen_step"]==0:
#         print("why?")
#     from transformers import AutoConfig, GemmaForCausalLM
#     model_id = "google/paligemma-3b-pt-224"
#     # 1. Fetch the full PaliGemma configuration
#     config = AutoConfig.from_pretrained(model_id)
#     # 2. Instantiate the Gemma language model using PaliGemma's text_config
#     #load barebones backbone (hidden states only):
#     hug_model = GemmaForCausalLM.from_pretrained(
#         model_id, 
#         config=config.text_config,
#         attn_implementation="eager",
#     )

# #my_GemmaModel initialization
#     config=my_GemmaConfig()
#     my_model=my_GemmaForCausalLM(config)
#     # copy values from pretrained model to my model
#     for my_key, my_value in my_model.state_dict().items():
#         match_tuple=[(hug_key, hug_param) for (hug_key, hug_param) in hug_model.state_dict().items() if my_key in hug_key]
#         match_key,match_value=match_tuple[0]
#         with torch.no_grad():
#             my_value.copy_(match_value)
#     # double check if copy successful torch.equal(torch.tensor([1, 2]), torch.tensor([1, 2]))
#     for my_key, my_value in my_model.state_dict().items():
#         match_tuple=[(hug_key, hug_param) for (hug_key, hug_param) in hug_model.state_dict().items() if my_key in hug_key]
#         match_key,match_value=match_tuple[0]
#         if not torch.equal(match_value,my_value):
#             print(my_value)
#             print(match_value)
#         assert torch.equal(match_value,my_value),f"{my_key}&{match_key} is not identical, the shape is {match_value.dim()}VS{my_value.dim()}"

# #prepare the inputs arguments for the model forward function
#     from transformers import AutoTokenizer
#     tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b")
#     prompt = "What is your favorite condiment?"
#     inputs = tokenizer(prompt, return_tensors="pt",padding="max_length",max_length=20)
#     input_ids=inputs.input_ids



# #run hg_model forward
#     hug_generate_ids = hug_model.generate(
#         input_ids, 
#         attention_mask=inputs.attention_mask,
#         max_length=23,
#         do_sample=True,
#         top_p=0.9,
#         top_k=0)
    

# #run my_model forward
#     hug_generate_ids=my_model.generate(
#         input_ids, 
#         attention_mask=inputs.attention_mask,
#         generation_len=23,
#         top_p=0.9)    


#     hg_tensor=utils.debug_values["debugging"]
#     my_tensor=debug_values["debugging"]
#     #check the value equality
#     print(f'value equality check________________________')
#     print(f'huggface:{hg_tensor}')
#     print(f'my_model:{my_tensor}')
#     bool_equal=torch.eq(hg_tensor, my_tensor)
#     print(bool_equal)
#     print(torch.all(bool_equal))  #for debugging 
   
#     # print the differences
#     # diff = (hg_tensor - my_tensor).abs()
#     # print(
#     #     f'max={diff.max().item():.8e}, '
#     #     f'mean={diff.mean().item():.8e}, '
#     # )

#     #check the shape equality
#     print(f'shape equality check________________________')
#     print(f'huggface:{hg_tensor.shape}')
#     print(f'my_model:{my_tensor.shape}')/???????????????????????????????????




