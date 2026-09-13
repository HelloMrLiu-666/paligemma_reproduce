@dataclasses
class my_PaliGemmaConfig():
    r"""
    Example:

    ```python
    >>> from transformers import PaliGemmaForConditionalGeneration, PaliGemmaConfig, SiglipVisionConfig, GemmaConfig

    >>> # Initializing a Siglip-like vision config
    >>> vision_config = SiglipVisionConfig()

    >>> # Initializing a PaliGemma config
    >>> text_config = GemmaConfig()

    >>> # Initializing a PaliGemma paligemma-3b-224 style configuration
    >>> configuration = PaliGemmaConfig(vision_config, text_config)

    >>> # Initializing a model from the paligemma-3b-224 style configuration
    >>> model = PaliGemmaForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "paligemma"
    vision_config: my_SiglipVisionConfig = None
    text_config: my_GemmaConfig = None
    image_token_index: int = 257152
    vocab_size: int = 257216
    projection_dim: int = 2048
    hidden_size: int = 2048
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        # BC: `use_bidirectional_attention` was originally unset in PaliGemma1 (backbone = Gemma1) AND PaliGemma2
        # (backbone = Gemma2). Both PaliGemmas want to default to True.
        if self.text_config.use_bidirectional_attention is None:
            self.text_config.use_bidirectional_attention = True

        self.text_config.num_image_tokens = (self.vision_config.image_size // self.vision_config.patch_size) ** 2

        self.vision_config.num_image_tokens=(self.vision_config.image_size // self.vision_config.patch_size) ** 2
        self.vision_config.projection_dim = self.projection_dim

        self.bos_token_id=self.text_config.bos_token_id
        self.eos_token_id=self.text_config.eos_token_id
        self.pad_token_id=self.text_config.pad_token_id
        self.ignore_index=-100
