from dataclasses import dataclass

@dataclass
class my_GemmaConfig():
    #following is the setting specific for paligemma 
    model_type = "gemma"
    keys_to_ignore_at_inference = ["past_key_values"]
    vocab_size: int = 257216
    hidden_size: int = 2048
    intermediate_size: int = 16384
    num_hidden_layers: int = 18
    num_attention_heads: int = 8
    num_key_value_heads: int = 1
    head_dim: int = 256
    hidden_act: str = "gelu_pytorch_tanh"
    max_position_embeddings: int = 8192
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    pad_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 1
    bos_token_id: int | None = 2
    tie_word_embeddings: bool = True
    # rope_parameters: RopeParameters | dict | None = None
    attention_bias: bool = False
    attention_dropout: float | int = 0.0
    use_bidirectional_attention: bool | None = None
            #"num_image_tokens": 256, #not gemma native ,is added in paligemma




