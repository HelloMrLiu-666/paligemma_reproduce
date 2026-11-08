from dataclasses import dataclass
from my_modeling_paligemma import PaliGemmaConfig
import torch
from torch import nn
from my_utils import _sample_top_p
from my_siglip.my_modeling_siglip import my_SiglipVisionModel
from my_gemma.my_modeling_gemma import my_GemmaModel
from my_gemma.my_cache_module import my_Cache


class PaliGemmaMultiModalProjector(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.linear =nn.Linear(
            config.vision_config.hidden_size, 
            config.text_config.hidden_size, 
            bias=True)

    def forward(self, image_features):
        """
        Args:
            image_features:(batch, num_posi,hidden_size_siglip)
        Returns:
            hidden_states: (batch, num_posi,hidden_size)
        """
        hidden_states = self.linear(image_features)
        return hidden_states


class PaliGemmaModel(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__(config)
        self.vision_tower = my_SiglipVisionModel(config.vision_config)
        self.multi_modal_projector = PaliGemmaMultiModalProjector(config)
        self.vocab_size = config.vocab_size

        self.language_model = my_GemmaModel(config.text_config)

        # self.text_config_dtype = 


    def get_image_features(
        self, pixel_values: torch.FloatTensor
    ) -> tuple[torch.FloatTensor,torch.FloatTensor]:
        #how the seq256 compress to seq1??: no need maybe to keep more seq would provide more image info to gemma
        """
        Args:
            pixel_values:(B,C,H,W)      PaliGemmaProcessor.prepare_inputs
        Returns:
            image_outputs: (batch, num_posi_image,hidden_size)
        """
        image_outputs,_=self.vision_tower(pixel_values) #(B,C,H,W)->[batch, num_positions,hidden_size_siglip] ,
        image_features=self.multi_modal_projector(last_hidden_state)#[batch, num_positions,hidden_size_siglip] ,[batch, num_positions,hidden_size]
        #image_features i.e. pooler_output
        return image_outputs,image_features

    def get_image_placeholder_mask(
        self, input_ids: torch.LongTensor, inputs_embeds: torch.FloatTensor, image_features: torch.FloatTensor
    ):
        """
        Obtains multimodal placeholder mask from `input_ids` or `inputs_embeds`, and checks that the placeholder token count is
        equal to the length of multimodal features. If the lengths are different, an error is raised.

        Args:
            input_ids:(batch, num_posi)
            inputs_embeds:(batch, num_posi,hidden_size)
            image_features:(batch, num_posi_image,hidden_size)
        Returns:
            special_image_mask: (batch, num_posi)
        """
        special_image_mask=(input_ids==config.image_token_index) #(batch, num_posi)->(batch, num_posi)
        img_count_mask=special_image_mask.sum().item()
        img_count_feature=image_features.shape[0]*image_features.shape[1]
        #check if image_features & special_image_mask match @ specific dimension
        if not (img_count_mask==img_count_feature):
            raise ValueError(f"image token count in image_features & special_image_mask not match! ")
        return special_image_mask

    def forward(
        self,
        input_ids: torch.LongTensor,        #provided by prepare_inputs_for_generation
        attention_mask: torch.Tensor,       #provided by prepare_inputs_for_generation
        position_ids: torch.LongTensor,     #provided by prepare_inputs_for_generation
        past_key_values: my_Cache,
        pixel_values: torch.FloatTensor | None = None, #should allow None to decide if merge image feature to text
        # token_type_ids: torch.LongTensor | None = None,
        # inputs_embeds: torch.FloatTensor | None = None,
        # labels: torch.LongTensor | None = None,
        # use_cache: bool | None = None,
        # **kwargs: Unpack[FlashAttentionKwargs],
    ) -> torch.Tensor:
        """
        Args:
            input_ids:(batch, num_posi1?)
            attention_mask:(num_posi1?, num_posi)
            position_ids:(batch, num_posi1?)
            past_key_values:Cache
            pixel_values:(B,C,H,W)
        Returns:
            last_hidden_state: (batch, num_posi1?,hidden_size)
        """
        # Replace image id with PAD if the image token if OOV, to avoid index-errors
        llm_input_ids = input_ids.clone()#(batch, num_posi1?)

        #no need to implement embedding layer within paligemma, just delegate inputs_embeds generation to gemma
        inputs_embeds=self.language_model.get_input_embeddings(llm_input_ids)  #(batch, num_posi1?)->(batch, num_posi1?,hidden_size)

        # Merge text and images, inputs_embeds & image_features, ony @ prefilling step
        if pixel_values is not None: #if none, means is not frefiling step, it's the auto-regressive generation step
            _,image_features = get_image_features(pixel_values) #(B,C,H,W)->(batch, num_posi_image,hidden_size)
            image_features=image_features.to(device=input_ids.device,dtype=inputs_embeds.dtype)
            special_image_mask=get_image_placeholder_mask(llm_input_ids, inputs_embeds, image_features).unsqueeze(-1)#(batch, num_posi1?)->(batch, num_posi1?,1)
            #(batch, num_posi1?,hidden_size),(batch, num_posi1?,1),(batch, num_posi_image,hidden_size)->(batch, num_posi1?,hidden_size)
            inputs_embeds=inputs_embeds.masked_scatter(special_image_mask, image_features)

        last_hidden_state,_ = self.language_modelforward(
                                    input_ids,
                                    attention_mask, 
                                    position_ids,
                                    past_key_values,
                                    inputs_embeds) 

        return last_hidden_state

class PaliGemmaForConditionalGeneration(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.config=config
        self.model = PaliGemmaModel(config)
        self.lm_head =nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def get_image_features(self, pixel_values: torch.FloatTensor):

        return 

    def forward(
        self,
        input_ids: torch.LongTensor, 
        attention_mask: torch.Tensor,
        position_ids: torch.LongTensor,
        past_key_values: Cache,
        pixel_values: torch.FloatTensor | None = None,#prefilling step should passed, auto-regressive generation step should None
        # token_type_ids: torch.LongTensor | None = None,
        # inputs_embeds: torch.FloatTensor | None = None,
        # labels: torch.LongTensor | None = None,
        # use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        # **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        """
        Args:
            input_ids:(batch, num_posi1?)
            attention_mask:(num_posi1?, num_posi)
            position_ids:(batch, num_posi1?)
            past_key_values:Cache
            pixel_values:(B,C,H,W)
        Returns:
            logits: (batch, num_posi1?,vocab_size)
        """
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.text_config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.text_config.vocab_size]`.

        Example:

        ```python
        >>> from PIL import Image
        >>> import httpx
        >>> from io import BytesIO
        >>> from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

        >>> model = PaliGemmaForConditionalGeneration.from_pretrained("google/paligemma2-3b-mix-224")
        >>> processor = AutoProcessor.from_pretrained("google/paligemma2-3b-mix-224")

        >>> prompt = "Where is the cat standing?"
        >>> url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
        >>> with httpx.stream("GET", url) as response:
        ...     image = Image.open(BytesIO(response.read()))

        >>> inputs = processor(images=image, text=prompt,  return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(**inputs,)
        >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Where is the cat standing?\nsnow"
        ```"""

        """
        Args:
            input_ids:(batch, num_posi1?)
            attention_mask:(num_posi1?, num_posi)
            position_ids:(batch, num_posi1?)
            past_key_values:Cache
            pixel_values:(B,C,H,W)
        Returns:
            logits: (batch, num_posi1?,vocab_size)
        """
    
        last_hidden_state = self.model( #(batch, num_posi1?,hidden_size)
            input_ids,
            attention_mask,
            position_ids,
            past_key_values,
            pixel_values) 

        #the aim of the slicing is to reduce the matrix operation at last projection layer
        #slice is feasible cause linear layer is point attention(1 posiiton only attend to itself's information), not self attention which attend to other positions information
        #for prefilling step the shape change is (batch, num_posi1?,hidden_size)->(batch, 1,hidden_size)
        #for the auto-regressive generation step, no shape change,
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        #(batch, num_posi1?,hidden_size)->(batch, num_posi1?,vocab_size)   
        logits = self.lm_head(last_hidden_state[:, slice_indices, :])
        return logits


    #essentiallly, huggingface implementation, the generation logic inherits from GenerationMxin class into PaliGemmaForConditionalGeneration
    #so, in my implementation, the generation logic also put within generate function and within PaliGemmaForConditionalGeneration class,
    #to mimic the usage of this the generation model as huggingface implementation, design this generate function
    #input_ids/attention_mask/position_ids are prepared in this function and passed to the forward function
    def generate( #the input of this function is the output of PaligemmaProcessor
        self,
        input_ids: torch.LongTensor, 
        attention_mask: torch.Tensor, #is the padding mask from the tokenizer
        pixel_values: torch.FloatTensor,
        generation_len: int | None = None,
        top_p: float):

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

            logits=self.forward(
                input_ids, 
                attention_mask,
                position_ids,
                past_key_values,
                pixel_values)
        #from logits to id in vocab, and update to the input_ids
            with torch.autocast(device_type=input_ids.device_type, enabled=False):
                #(batch, num_posi1?,vocab_size) no shape change
                output_probs=nn.functional.softmax(logits, dim=-1, dtype=torch.float32)
            #(batch, num_posi1?,vocab_size)->(batch, num_posi1?)
            next_token_id=_sample_top_p(output_probs,top_p)
            #update the input_ids to prepare it for the auto-regressive generation
            input_ids=next_token_id
            #fill the next_token_id to generated_ids ,(b,generation_len), (batch, num_posi1?)
            generated_ids[:,i]=next_token_id

        return generated_ids

            


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
