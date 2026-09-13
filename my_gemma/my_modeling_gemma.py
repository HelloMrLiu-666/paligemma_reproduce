from typing import Callable
import torch
import torch.nn as nn
from my_configuration_gemma import my_GemmaConfig
from my_cache_module import my_Cache
from transformers import AutoTokenizer, GemmaForCausalLM


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
        self.weight = nn.Parameter(data=torch.ones(*(1,1,dim),  dtype=torch.float32))

    def _norm(self, x):
        """
        Args:
            x:[batch, num_posi,emb_dim]
        Returns:
            normalized:[batch, num_posi,1]
        """
        x=x.power(2).mean(dim=-1, keepdim=True)#[batch, num_posi,emb_dim]->[batch, num_posi,1]
        normal_factor=torch.rsqrt(x+self.eps)#no shape change
        return normal_factor

    def forward(self, x):
        with torch.autocast(device_type=x.device_type, enabled=False):
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
        #to device is not set for the tensor here, so the device need to be set within forward before operation with GPU tensor x
        #otherwise the device should be desiged as a separate argument to pass through multiple functions to get to here
        exponents=torch.arange(0,dim,2, dtype=torch.float)#even though looks like not the same as paper, the final range of the exponents is the same as the paper [0,dim-2]
        theta=10000.0**(-exponents/dim.float())
        return theta


    def forward(self, x, position_ids):
        """
        Args:
            x: [batch,heads, num_posi,head_dim]
            position_ids:(1,num_posi_1?)

        Returns:
            cos : [len=1?,head_dim]
            sin : [len=1?,head_dim]
        """
        #my implementation is different to huggingface, I haven't include batch information in cos & sin
        position_ids_sliced=(position_ids[0,:])[:,None]#(1,len=1?)->(len=1?,)->(len=1?,1)
        theta=(self.inv_freq)[None,:].to(device=x.device)#(head_dim/2,)->(1,head_dim/2)
        with torch.autocast(device_type=x.device.type, enabled=False):
            m_theta=torch.matmul(position_ids_sliced.float(), theta.float())#(len=1?,1)@(1,head_dim/2)->(len=1?,head_dim/2)
            cos=torch.cos(m_theta)
            sin=torch.sin(m_theta)
            cos=torch.cat((cos, cos), dim=1)#(len=1?,head_dim/2)->(len=1?,head_dim)
            sin=torch.cat((sin, sin), dim=1)
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
    x2=-x[...,head_dim//2:]

    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): (batch,heads, len=1?,head_dim)
        k (`torch.Tensor`): (batch,heads, len=1?,head_dim)
        cos (`torch.Tensor`): (len=1?,head_dim)
        sin (`torch.Tensor`): (len=1?,head_dim)
        unsqueeze_dim (`int`, *optional*, defaults to 1):
    Returns:
        q_embed: (batch,heads, len=1?,head_dim)
        k_embed: (batch,heads, len=1?,head_dim)
    """
    q_rotate_half=rotate_half(q)
    k_rotate_half=rotate_half(k)    #(batch,heads, len=1?,head_dim)
    cos=cos[None,None,:,:]#(len=1?,head_dim)->(1,1,len=1?,head_dim)
    sin=sin[None,None,:,:]
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
    #(batch,att_heads,num_posi_query,head_dim)@(batch,att_heads,head_dim,num_posi_key)->(batch,att_heads,num_posi_query,num_posi_key)
    attn_weights=torch.matmul(query, key_states.transpose(2, 3))*scaling
    if not attention_mask is None:
        attn_weights=attn_weights+attention_mask

    with torch.autocast(device_type=query.device.type, enabled=False):
        attn_weights=nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
    attn_weights=attn_weights.to(query.dtype)#only the softmax is calculated as float32, other matrix multiplication might in lower precision
    attn_weights=nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    #(batch,att_heads,num_posi_query,num_posi_key)@(batch,att_heads,num_posi_key,head_dim)->(batch,att_heads,num_posi_query,head_dim)
    attn_output=torch.matmul(attn_weights, value_states)

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
        if past_key_values.get_seq_length() > 0:#(batch,num_kv_heads,num_posi_1?,head_dim)->(batch,num_kv_heads,num_posi,head_dim)
            k_states, v_states=past_key_values.update(k_states, v_states, self.layer_idx)

    #align the heads
        k_states=repeat_kv(k_states,self.num_key_value_groups)
        v_states=repeat_kv(v_states,self.num_key_value_groups)

        attn_output, attn_weights=eager_attention_forward(#no shape change for q_state (batch,num_att_heads,num_posi_1?,head_dim)
            module=self,
            query=q_states,
            key=k_states,
            value=v_states,
            attention_mask=attention_mask,
            scaling=self.scaling,
            dropout=self.attention_dropout)
        #(batch,num_att_heads,num_posi_1?,head_dim)->(batch,num_posi_1?,num_att_heads,head_dim)->(batch,num_posi_1?,num_att_heads*head_dim)
        attn_output=attn_output.transpose(1,2).contiguous().view(*shape,-1)
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

        for layer in self.layers:
            hidden_states=layer(
                hidden_states,
                attention_mask,
                position_ids,
                past_key_values,
                position_embeddings
            )
        
        hidden_states=self.norm(hidden_states)#no shape change\
        last_hidden_state=hidden_states

        return last_hidden_state



class my_GemmaForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
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

        r"""
        Example:

        ```python
        >>> from transformers import AutoTokenizer, GemmaForCausalLM

        >>> model = GemmaForCausalLM.from_pretrained("google/gemma-7b")
        >>> tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b")

        >>> prompt = "What is your favorite condiment?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "What is your favorite condiment?"
        ```"""
        hidden_states=self.model(input_ids,attention_mask,past_key_values)[0]
        logits=self.lm_head(hidden_states)#(batch, num_posi_1?,emb_dim)->(batch, num_posi_1?,vocab)
        return logits

    #essentiallly, huggingface implementation, the generation logic inherits from GenerationMxin class into PaliGemmaForConditionalGeneration
    #so, in my implementation, the generation logic also put within generate function and within PaliGemmaForConditionalGeneration class,
    #to mimic the usage of this the generation model as huggingface implementation, design this generate function
    #input_ids/attention_mask/position_ids are prepared in this function and passed to the forward function
    def generate( #the input of this function is the output of PaligemmaProcessor
        self,
        input_ids: torch.LongTensor, 
        attention_mask: torch.Tensor, #is the padding mask from the tokenizer
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
                #initialize a padding tensor of shape (b,generation_len)
                generated_ids=torch.full((input_ids.shape[0],generation_len), pad_token_id, dtype=input_ids.dtype,device=input_ids.device)
            else:
                is_first_iteration=False

            #each generation step, updates the attention_mask,position_ids 
            attention_mask,position_ids=self.prepare_inputs_for_generation(#the returned mask is the paligemma specific mask
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
            with torch.autocast(device_type=input_ids.device_type, enabled=False):
                #(batch, num_posi1?,vocab_size) no shape change
                output_probs=nn.functional.softmax(logits, dim=-1, dtype=torch.float32)
            #(batch, num_posi1?,vocab_size)->(batch, num_posi1?)
            next_token_id=self.my_sample_top_p(output_probs,top_p)
            #update the input_ids to prepare it for the auto-regressive generation
            input_ids=next_token_id
            #fill the next_token_id to generated_ids ,(b,generation_len), (batch, num_posi1?)
            generated_ids[:,i]=next_token_id

        return generated_ids

            
    def my_sample_top_p(probs, p):# (b,seq,vocab)->(b,seq,1)
        b,s,v=probs.shape
        # g_enerator = torch.Generator(device=probs.get_device())
        probs, indices=torch.sort(probs, dim=-1, descending=True)# indices are the index in unsorted probs
        probs_sum=torch.cumsum(probs, dim=-1) #(b,seq,vocab)
        mask=probs_sum-probs >p
        #keep the top P
        probs[mask]=0.0 #utilizing the boolean indexing
        #normalize the top P
        probs=probs/torch.sum(probs, dim=-1, keepdim=True)#(b,seq,vocab)/(b,seq,1)->(b,seq,vocab)
        #sample from the top P, recent version pytorch support 3D input tensor for multinomial, the last dim is the sampling dim
        probs=probs.view(-1,v)
        probs_index=torch.multinomial(probs, num_samples=1, generator=None)#(b*seq,vocab)->index (b*seq,1) 
        probs_index=probs_index.view(b,s,1)#the multinomial is sandwiched because multinomial support upto 2D
        
        next_token=torch.gather(indices, dim=-1,index=probs_index)#(b,seq,vocab),index (b,seq,1) ->(b,seq,1)
        return next_token

    def prepare_inputs_for_generation(
                self,
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
            attention_mask:(batch,num_posi1?, num_posi)
            position_ids:(1, num_posi1?)
        """
        batch,q_len,_=input_ids.shape
        # position_ids
        if is_first_iteration:
            position_ids=torch.arange(start=0, end=input_ids.shape[1], dtype=torch.Long, device=input_ids.device)
            position_ids=position_ids[None,:]#(num_posi_1?,)->(1,num_posi_1?)
        else:
            #prepare the position_id
            old_len=past_key_values.get_seq_length(layer_idx=0)#before/after decoding layer operation, all layer should have same seq_len
            new_len_1=q_len
            position_ids=torch.arange(start=old_len, end=old_len+new_len_1, dtype=torch.Long, device=input_ids.device)
            position_ids=position_ids[None,:]#(num_posi_1?,)->(1,num_posi_1?)
        position_ids=position_ids+1#in Paligemma are 1-indexed

        # Create the mask, only works for no padding existed in the input_ids
        if is_first_iteration:
            attention_mask=torch.full((batch,q_len,q_len), 0.0, dtype=self.lm_head.dtype,device=input_ids.device)
        else:
            assert q_len == 1
            kv_len =past_key_values.get_seq_length(layer_idx=0) + q_len
            attention_mask=torch.full((batch,q_len,kv_len), 0.0, dtype=self.lm_head.dtype,device=input_ids.device)

        return attention_mask,position_ids


def test():
# if __name__ == "__main__":
    
    prompt = "What is your favorite condiment?"
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b")
    inputs = tokenizer(prompt, return_tensors="pt")

#huggingface model generation
    # hug_model = GemmaForCausalLM.from_pretrained("google/gemma-7b")
    # # Generate
    # hug_generate_ids = hug_model.generate(inputs.input_ids, max_length=30)
    # hug_output_text=tokenizer.batch_decode(hug_generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

#my model generation
    config=my_GemmaConfig()
    print("hahha")
    with torch.device('meta'):   
        my_model=my_GemmaForCausalLM(config)
    print("ee")

    num_params = sum(p.numel() for p in my_model.parameters())
    print(f"Parameters: {num_params:,}")
    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        bytes_per_param = torch.tensor([], dtype=dtype).element_size()

        memory_gb = (
            num_params * bytes_per_param / 1024**3
        )

        print(dtype, f"{memory_gb:.2f} GB")
    #prepare the attention_mask

    # Generate
    my_generated_ids=my_model.generate( 
            inputs.input_ids, 
            None, #to simplify the implementation, assuming that no padding(attention mask) included
            generation_len= 1,
            top_p=0.9)
    my_output_text=tokenizer.batch_decode(my_generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    #compare model output
    print(f'my_output_text:{my_output_text}')
    # print(f'hug_output_text:{hug_output_text}')