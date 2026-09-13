import numpy as np
import PIL.Image
from transformers import GemmaTokenizer
# from  my_image_process_siglip import ImageProcessorPil
import httpx
from io import BytesIO

from ..my_siglip.my_image_process_siglip import ImageProcessorPil


IMAGE_TOKEN = "<image>"


#prepare image and text data for the tokenizer, 
#the tokenizer would further process the PaliGemmaProcessor's outputs to get the input_ids
class PaliGemmaProcessor():  
    def __init__(
        self,
        image_processor:my_ImageProcessorPil,
        tokenizer,
        image_seq_length:int,
        config:my_PaliGemmaConfig,
    ):
        # if not hasattr(image_processor, "image_seq_length"):
        #     raise ValueError("Image processor is missing an `image_seq_length` attribute.")

        self.image_seq_length = image_seq_length
        self.config=config

        if not hasattr(tokenizer, "image_token"):
            # image_token = AddedToken(IMAGE_TOKEN, normalized=False, special=True)
            # tokens_to_add = {"additional_special_tokens": [image_token]}
            # tokenizer.add_special_tokens(tokens_to_add)
            special_tokens_dict = {"additional_special_tokens": IMAGE_TOKEN}
            num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
            self.image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
            self.image_token = IMAGE_TOKEN
        else:
            self.image_token_id = tokenizer.image_token_id
            self.image_token = tokenizer.image_token

        # tokenizer.add_tokens(EXTRA_TOKENS)
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False

    def prepare_inputs(
            self, 
            images:list[list[PIL.Image]],  # type: ignore
            text:list[str]=None)->tuple[list[list[PIL.Image]],list[str]]:
        """
        Args:
            images: list of list of images(each prompt could corresponding to list of images)
            text:list of strings
        Returns:
            pixel_values: (B,C,H,W) -- Pixel values to be fed to a model. Returned when `images` is not `None`.
            input_ids: (B,seq) -- input ids with image placeholder
        """
        if (
            isinstance(images,list)
            and isinstance(images[0], PIL.Image)#to simplify the implementation, each prompt corresponding to !!single!!! image
        ):
            text = [
                f"{IMAGE_TOKEN * self.image_seq_length}{self.tokenizer.bos_token}{prompt}\n"
                for prompt, image_list in zip(text, images)
            ]
            #
            image_size=self.config.vision_config.image_size #224
            pixel_values=image_processor.process_images(images,(image_size, image_size),PILImageResampling.BICUBIC) #(B,C,H,W)
            
            encoded = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
            input_ids=encoded["input_ids"]
            attention_mask=encoded["attention_mask"] #this attention_mask is corresponding to the automated padding due to the tokenizer

        return input_ids,attention_mask,pixel_values
    


if __name__ == "__main__":

    tokenizer= GemmaTokenizer.from_pretrained("google/paligemma-3b-pt-224")
    image_processor=ImageProcessorPil()

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    with httpx.stream("GET", url) as response:
        image = PIL.Image.open(BytesIO(response.read()))


    images=[list[image]]
    text=["hello"]
    paliGemma_processor=PaliGemmaProcessor(tokenizer,256)
    paliGemma_processor.prepare_inputs(images,text)



