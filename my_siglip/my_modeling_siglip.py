
from io import BytesIO
import httpx
import torch
import torch.nn as nn
from transformers import SiglipVisionModel,AutoProcessor
from my_configuration_siglip import my_SiglipVisionConfig
import PIL.Image
from my_image_process_siglip import my_ImageProcessorPil
import  transformers





PILImageResampling = PIL.Image.Resampling

#this is for liu_guang debugging
my_debug_values = {}

class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config: my_SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels, 
            out_channels=self.embed_dim, 
            kernel_size=self.patch_size, 
            stride=self.patch_size, 
            padding="valid")

        self.num_patches = (self.image_size//self.patch_size)**2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, #size of the dictionary 
                                               self.embed_dim)
        self.register_buffer("position_ids",
                             (torch.arange(self.num_positions)).expand(1, -1), #[1,num_positions]
                             persistent=False)  #no need to specify the device, because it's registered buffer

        

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        """
        pixel_values: of shape [batch,channel, height, width]

        Returns:
            `tensor`: of shape [batch, num_positions,embed_dim]
        """
        weight_type=self.patch_embedding.weight.dtype
        batch,channel, height, width=pixel_values.shape

        #[batch,channel, height, width]->[batch,embed_dim, height/patch_size, width/patch_size]
        hidden_dim=self.patch_embedding(pixel_values.to(dtype=weight_type))

        #[batch,embed_dim, height/patch_size, width/patch_size]->[batch,embed_dim, num_positions]
        hidden_dim=hidden_dim.contiguous().view(batch,self.embed_dim,self.num_positions)
        #[batch,embed_dim, num_positions]->[batch, num_positions,embed_dim]
        hidden_dim=hidden_dim.contiguous().transpose(1,2)
        #[1,num_positions]->#[1,num_positions,embed_dim]
        position_ids=self.position_embedding(self.position_ids)
        #[batch, num_positions,embed_dim]+[1,num_positions,embed_dim]->[batch, num_positions,embed_dim]
        embeddings=hidden_dim+position_ids

        return embeddings


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    nth_layer=0
)-> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Args:
        query:[batch,num_heads, num_posi,head_dim]
        key:[batch,num_heads, num_posi,head_dim]
        value:[batch,num_heads, num_posi,head_dim]
    Returns:
        attn_output: [batch,num_heads, num_posi,head_dim]
        attn_weights:[batch,num_heads,num_posi, num_posi]
    """
    #[batch,num_heads, num_posi,head_dim] *[batch,num_heads,head_dim, num_posi]->[batch,num_heads,num_posi, num_posi]

    #for benchmarking debugging 
    if(nth_layer==0):
        my_debug_values["debugging"] =query.detach().clone() #hidden_dim.detach().clone()

    attn_weights=torch.matmul(query,key.transpose(2,3))*scaling
    if attention_mask==None:
        attention_mask=torch.zeros_like(attn_weights)#[batch,num_heads,num_posi, num_posi]
    attn_weights=attn_weights+attention_mask
    attn_weights=nn.functional.softmax(input=attn_weights, dim=-1, dtype=torch.float32).to(dtype=query.dtype)
    attn_weights=nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    #[batch,num_heads,num_posi, num_posi]*[batch,num_heads, num_posi,head_dim]->[batch,num_heads, num_posi,head_dim]
    attn_output=torch.matmul(attn_weights,value)
    return attn_output, attn_weights



class SiglipAttention(nn.Module):
    def __init__(self, config:my_SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim//self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError("head_dim*num_heads should be equal to embed_dim")
        self.scale = (self.head_dim)**(-0.5)
        self.dropout = config.attention_dropout
        self.training=self.training
        self.is_causal = False

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        nth_layer=0, #just for debugging
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Rescales `image` by `scale`.

        Args:
            hidden_states:[batch, num_positions,embed_dim]
            attention_mask:
        Returns:
            attn_output: [batch, num_posi,embed_dim]
            attn_weights:[batch,num_heads,num_posi, num_posi]
        """
        batch, num_posi,embed_dim=hidden_states.shape
        hidden_shape=(batch, num_posi,self.num_heads,-1)#[batch, num_posi,num_heads,head_dim]
        q=self.q_proj(hidden_states)#[batch, num_posi,embed_dim]->[batch, num_posi,embed_dim]
        k=self.k_proj(hidden_states)
        v=self.v_proj(hidden_states)
        #[batch, num_posi,embed_dim]->[batch, num_posi,num_heads,head_dim]->[batch,num_heads, num_posi,head_dim]
        q=q.view(hidden_shape).transpose(1,2)
        k=k.view(hidden_shape).transpose(1,2)
        v=v.view(hidden_shape).transpose(1,2)
        attn_output, attn_weights=eager_attention_forward(
            self,
            query=q,
            key=k,
            value=v,
            attention_mask=None,
            scaling=self.scale,
            dropout=self.dropout,
            nth_layer=nth_layer
            )
        #[batch,num_heads, num_posi,head_dim]->#[batch, num_posi,num_heads,head_dim]->#[batch, num_posi,embed_dim]
        attn_output=attn_output.transpose(1,2).contiguous().view(batch, num_posi,-1)



        attn_output=self.out_proj(attn_output)#[batch, num_posi,embed_dim] no shape change
        return attn_output, attn_weights

class SiglipMLP(nn.Module):
    def __init__(self, config:my_SiglipVisionConfig):
        super().__init__()
        self.config = config
        # self.activation_fn = ACT2FN[config.hidden_act]
        # self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.fc2 = nn.Linear(config.intermediate_size,config.hidden_size,  bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states:[batch, num_positions,embed_dim]
        Returns:
            hidden_states:[batch, num_positions,embed_dim]
        """
        hidden_states=nn.functional.gelu(self.fc1(hidden_states), approximate='tanh')
        hidden_states=self.fc2(hidden_states)
        return hidden_states


class SiglipEncoderLayer(nn.Module):
    def __init__(self, config: my_SiglipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps, bias=True)
        self.self_attn = SiglipAttention(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps, bias=True)
        self.mlp = SiglipMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        nth_layer=0 #just for debugging
    ) -> torch.FloatTensor:
        """
        Args:
            hidden_states:[batch, num_positions,embed_dim]
        Returns:
            hidden_states:[batch, num_positions,embed_dim]
        """
        residual=hidden_states
        hidden_states=self.layer_norm1(hidden_states)
        hidden_states,_=self.self_attn(hidden_states,
                                       attention_mask,
                                       nth_layer#just for debugging
                                       )

        hidden_states=hidden_states+residual

        residual=hidden_states
        hidden_states=self.layer_norm2(hidden_states)
        hidden_states=self.mlp(hidden_states)
        hidden_states=hidden_states+residual
            
        return hidden_states

class SiglipEncoder(nn.Module):
    def __init__(self, config: my_SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([SiglipEncoderLayer(config) for i in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    # Ignore copy
    def forward(
        self,
        inputs_embeds,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            inputs_embeds:[batch, num_positions,embed_dim]
        Returns:
            hidden_states:[batch, num_positions,embed_dim]
        """
        hidden_states=inputs_embeds

        nth_layer=0#just for debugging
        for layer in self.layers:
            hidden_states=layer(hidden_states,
                                attention_mask,
                                nth_layer=nth_layer#just for debugging
                                )
            nth_layer=nth_layer+1#just for debugging

        return hidden_states


class my_SiglipVisionModel(nn.Module):
    def __init__(self, config: my_SiglipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm =  nn.LayerNorm(embed_dim, eps=config.layer_norm_eps, bias=True)
        self.use_head = True if not hasattr(config, "vision_use_head") else config.vision_use_head
        self.vision_use_head=config.vision_use_head
        if self.use_head:
            self.head = SiglipMultiheadAttentionPoolingHead(config)

    def forward(
        self,
        pixel_values,
    )  -> tuple[torch.Tensor,torch.Tensor]:
        """
        Args:
            pixel_values: of shape [batch,channel, height, width]
        Returns:
            last_hidden_state:[batch, num_positions,embed_dim]
            pooler_output:[batch, embed_dim]|None

        """

        
        r"""
        Examples from huggingface standard implementation:

        ```python
        >>> import httpx
        >>> from io import BytesIO
        >>> from PIL import Image
        >>> from transformers import AutoProcessor, SiglipVisionModel

        >>> model = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-224")
        >>> processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> with httpx.stream("GET", url) as response:
        ...     image = Image.open(BytesIO(response.read()))

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled features
        ```"""



        hidden_states=self.embeddings(pixel_values)#[batch,channel, height, width]->[batch, num_positions,embed_dim]

        encoder_outputs=self.encoder(hidden_states)#no shape change [batch, num_positions,embed_dim]



        last_hidden_state =encoder_outputs
        last_hidden_state=self.post_layernorm(last_hidden_state)#no shape change [batch, num_positions,embed_dim]
        if self.vision_use_head:
            pooler_output=self.head(last_hidden_state)#[batch, num_positions,embed_dim]->[batch, embed_dim]
        else:
            pooler_output=None
        return last_hidden_state,pooler_output


class SiglipMultiheadAttentionPoolingHead(nn.Module):
    """Multihead Attention Pooling."""

    def __init__(self, config: my_SiglipVisionConfig):
        super().__init__()

        self.probe = nn.parameter.Parameter(torch.randn(1,1, config.hidden_size))
        self.attention = nn.MultiheadAttention(config.hidden_size, config.num_attention_heads,batch_first= True)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, bias=True)
        self.mlp = SiglipMLP(config)

    def forward(self, hidden_state):
        """
        Args:
            hidden_state:[batch, num_positions,hidden_size]
        Returns:
            hidden_state[:, 0]:[batch, hidden_size]
        """
        batch=hidden_state.shape[0]
        query=self.probe.expand(batch,-1, -1)#[1,1,hidden_size]->[batch,1,hidden_size]
        #[batch,1,hidden_size]&[batch,num_positions,hidden_size]->[batch,1,hidden_size]
        hidden_state, _ = eager_attention_forward(
            query=query,
            key=hidden_state, 
            value=hidden_state)[0]
        residual=hidden_state
        hidden_state=self.layernorm(hidden_state)
        hidden_state=self.mlp(hidden_state)
        hidden_state=residual+hidden_state

        return hidden_state[:, 0]


if __name__ == "__main__":


    torch.manual_seed(100)

    print(transformers.__file__)
#processor to prepare the inputs
    from pathlib import Path
    import transformers.image_processing_backends as image_processing_backends #import TorchvisionBackend

    image_path = Path(__file__).parent / "000000039769.jpg"
    image = PIL.Image.open(image_path)

    #huggingface processor
    processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
    hug_inputs = processor(images=image, return_tensors="pt")

    #my processor
    image_processor=my_ImageProcessorPil()
    my_inputs=image_processor.process_images(
        images =[image],
        size = (224, 224),
        resample =PILImageResampling.BICUBIC
        )
    
    #compare my_inputs & hug_inputs
    # print(f'my_inputs:{my_inputs}')
    # print(f'hug_inputs:{hug_inputs["pixel_values"]}')
    bool_equal=torch.eq(my_inputs, hug_inputs["pixel_values"])
    # print(bool_equal)
    print(torch.all(bool_equal))

#compare the model output
    config=my_SiglipVisionConfig()
    #huggingface model
    hug_model = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-224",attn_implementation="eager")
    import inspect
    print(inspect.getfile(hug_model.__class__))
    # hug_outputs = hug_model(**hug_inputs)
    # hug_last_hidden_state = hug_outputs.last_hidden_state
    # hug_pooled_output = hug_outputs.pooler_output  # pooled features

    #my SiglipVisionModel model
    my_model=my_SiglipVisionModel(config)

    # copy values from pretrained model to my model
    for my_key, my_value in my_model.state_dict().items():
        match_tuple=[(hug_key, hug_param) for (hug_key, hug_param) in hug_model.state_dict().items() if my_key in hug_key]
        match_key,match_value=match_tuple[0]
        with torch.no_grad():
            my_value.copy_(match_value)

    # double check if copy successful torch.equal(torch.tensor([1, 2]), torch.tensor([1, 2]))
    for my_key, my_value in my_model.state_dict().items():
        match_tuple=[(hug_key, hug_param) for (hug_key, hug_param) in hug_model.state_dict().items() if my_key in hug_key]
        match_key,match_value=match_tuple[0]
        if not torch.equal(match_value,my_value):
            print(my_value)
            print(match_value)
        assert torch.equal(match_value,my_value),f"{my_key}&{match_key} is not identical, the shape is {match_value.dim()}VS{my_value.dim()}"

    hug_outputs = hug_model(**hug_inputs)
    hug_last_hidden_state = hug_outputs.last_hidden_state

    my_outputs,_=my_model(my_inputs)#[batch=1,channel, height, width]->[batch, embed_dim]   

    #for benchmarking debugging 
    import transformers.models.siglip.modeling_siglip as modeling_siglip
    hg_tensor=modeling_siglip.debug_values["debugging"]
    my_tensor=my_debug_values["debugging"]
    print(f'huggingface:{hg_tensor}')
    print(f'my_model:{my_tensor}')
    bool_equal=torch.eq(hg_tensor, my_tensor)
    print(bool_equal)
    print(torch.all(bool_equal))  #for debugging 
    # print the differences
    diff = (hg_tensor - my_tensor).abs()
    print(
        f'max={diff.max().item():.8e}, '
        f'mean={diff.mean().item():.8e}, '
    )



    #compare model output
    print(f'my_outputs:{my_outputs}')
    print(f'hug_last_hidden_state:{hug_last_hidden_state}')
    print(torch.all(torch.eq(my_outputs, hug_last_hidden_state)))
    diff = (my_outputs - hug_last_hidden_state).abs()
    print(
        f'max={diff.max().item():.8e}, '
        f'mean={diff.mean().item():.8e}, '
    )




