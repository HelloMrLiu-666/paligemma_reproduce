
from dataclasses import dataclass

##following is paligemma checkpoint specific
# @dataclass
# class my_SiglipVisionConfig():
#     model_type = "siglip_vision_model"
#     base_config_key = "vision_config"

#     hidden_size: int = 1152
#     intermediate_size: int = 4304
#     num_hidden_layers: int = 27
#     num_attention_heads: int = 16
#     num_channels: int = 3
#     image_size: int | list[int] | tuple[int, int] = 224
#     patch_size: int | list[int] | tuple[int, int] = 14
#     hidden_act: str = "gelu_pytorch_tanh"
#     layer_norm_eps: float = 1e-6
#     attention_dropout: float | int = 0.0
#     vision_use_head:bool= False
#     #following is additional attributes from paligemma
#     # "num_image_tokens": 256,  #not siglip native ,is added in paligemma
#     # "projection_dim": 2048,  #not siglip native ,is added in paligemma


#following is siglip checkpoint specific
class my_SiglipVisionConfig():
    model_type = "siglip_vision_model"
    base_config_key = "vision_config"

    hidden_size: int = 768
    intermediate_size: int = 3072
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    num_channels: int = 3
    image_size: int | list[int] | tuple[int, int] = 224
    patch_size: int | list[int] | tuple[int, int] = 16
    hidden_act: str = "gelu_pytorch_tanh"
    layer_norm_eps: float = 1e-6
    attention_dropout: float | int = 0.0

    vision_use_head:bool= False

