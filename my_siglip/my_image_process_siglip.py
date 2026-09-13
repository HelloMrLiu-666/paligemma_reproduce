from io import BytesIO
from pathlib import Path
from typing import Collection, Optional
import httpx
import numpy as np
import PIL.Image
from transformers.image_utils import ImageInput
from torchvision.transforms.v2 import functional as tvF

import torch
from transformers import AutoProcessor

from torchvision.transforms import InterpolationMode

PILImageResampling = PIL.Image.Resampling


#this is for liu_guang debugging
debug_values_my = {}

def resize(
    image: ImageInput,
    size: tuple[int, int],
    resample: Optional[PIL.Image.Resampling] = None,
    reducing_gap: int | None = None,
) -> PIL.Image:
    """
    Resizes `image` to `(height, width)` specified by `size` using the PIL library.

    Args:
        image (`PIL.Image`):
            The image to resize.
        size (`tuple[int, int]`):
            The size to use for resizing the image.
        resample (`int`, *optional*, defaults to `PILImageResampling.BILINEAR`):
            The filter to user for resampling.
        reducing_gap (`int`, *optional*):
            Apply optimization by resizing the image in two steps. The bigger `reducing_gap`, the closer the result to
            the fair resampling. See corresponding Pillow documentation for more details.
    Returns:
        `PIL.Image`: The resized image.
    """

    if not len(size) == 2:
        raise ValueError("The size dimension of the image should be 2")
    # resized_img=image.resize(size=size, resample=resample, reducing_gap= reducing_gap) 
    resized_img=tvF.resize(image, size, interpolation=resample, antialias=True)
    

    return  resized_img

def rescale(
    image: np.ndarray,
    scale: float,
    dtype: torch.dtype = torch.float32,
) -> np.ndarray:
    """
    Rescales `image` by `scale`.

    Args:
        image (`np.ndarray`):
            The image to rescale.
        scale (`float`):
            The scale to use for rescaling the image.
        dtype (`np.dtype`, *optional*, defaults to `np.float32`):
            The dtype of the output image. Defaults to `np.float32`. Used for backwards compatibility with feature
            extractors.
    Returns:
        `np.ndarray`: The rescaled image.
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input image must be of type torch.Tensor, got {type(image)}")
    
    image=(image.to(torch.float64))*scale
    return image.to(dtype)

def normalize(
    image: torch.Tensor,
    mean: float | Collection[float],
    std: float | Collection[float],
) -> np.ndarray:
    """
    Normalizes `image` using the mean and standard deviation specified by `mean` and `std`.

    image = (image - mean) / std

    Args:
        image (`np.ndarray`):
            The image to normalize.
        mean (`float` or `Collection[float]`):
            The mean to use for normalization.
        std (`float` or `Collection[float]`):
            The standard deviation to use for normalization.
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError("image must be a torch.Tensor")
    
    # mean=(torch.tensor(mean, dtype=torch.float32))[:, None, None]#shape from (3,) to (3,1,1) #channel first
    # std=(torch.tensor(std, dtype=torch.float32))[:, None, None]
    # image = (image.to(torch.float32) - mean) / std
    #following is the implementation method from huggingface 
    mean=mean*255.0
    std=std*255.0
    image =tvF.normalize(image.to(dtype=torch.float32), mean.to(dtype=torch.float32), std.to(dtype=torch.float32))
    return image

def convert_to_rgb(image: PIL.Image) -> PIL.Image:
    """
    Converts an image to RGB format. Only converts if the image is of type PIL.Image.Image, otherwise returns the image
    as is.
    Args:
        image (PIL.Image):
            The image to convert.
    """
    if image.mode == "RGB":
        return image

    image = image.convert("RGB")
    return image


class my_ImageProcessorPil():
    def __init__(
        self, 
        IMAGENET_STANDARD_MEAN:torch.Tensor= torch.tensor([0.5, 0.5, 0.5]),
        IMAGENET_STANDARD_STD:torch.Tensor= torch.tensor([0.5, 0.5, 0.5]),
        )-> None:
        self.IMAGENET_STANDARD_MEAN=IMAGENET_STANDARD_MEAN
        self.IMAGENET_STANDARD_STD =IMAGENET_STANDARD_STD

    def process_images(
        self,
        images:list[ImageInput]=None,
        size:tuple[int, int] = (224, 224),
        resample =InterpolationMode.BICUBIC
        ):
        """
        convert_to_rgb->resize->Convert to NumPy array->rescale->normalize->Convert to tensor

        Returns:
            `tensor`: channel first tensor. (B,C,H,W) ie channel first
        """

        #using for loop to process image data, low efficient, just to simplify my implementation; 
        #future improvement could support batch tensor, need to improve every submethod to support batch tensor
        image_list=[]
        for image in images:
            #convert from channel last to channel first
            image=convert_to_rgb(image=image)
            if not image.mode=='RGB':
                raise TypeError("It's not a RGB image")
            # Convert to NumPy array,
            image = tvF.pil_to_tensor(image)

            ## Convert to NumPy array, and reshape to (C,H,W) ie channel first
            # image = np.asarray(image)
            if image.shape[0]==3:
                print(f"the image is channel first {image.shape}")
            elif image.shape[2]==3:
                image=image.permute(2, 0, 1).contiguous()
                print(f"channel last image reshape to {image.shape}")
            #resize the image 
            image = resize(image=image, size=size, resample=resample)
            #normalize the image
            image = normalize(image, self.IMAGENET_STANDARD_MEAN, self.IMAGENET_STANDARD_STD)
            # #for debugging 
            # debug_values_my["debugging"] = image
            # hg_tensor=image_processing_backends.debug_values["debugging"]
            # my_tensor=debug_values_my["debugging"]
            # print(f'huggingface:{hg_tensor}')
            # print(f'my_model:{my_tensor}')
            # bool_equal=torch.eq(hg_tensor, my_tensor)
            # print(bool_equal)
            # print(torch.all(bool_equal))     

            image_list.append(image)
        return torch.stack(image_list, dim=0) #(C,H,W)->(B,C,H,W)

# if __name__ == "__main__":
#     from pathlib import Path
#     import transformers.image_processing_backends as image_processing_backends #import TorchvisionBackend

#     image_path = Path(__file__).parent / "000000039769.jpg"
#     image = PIL.Image.open(image_path)

#     #huggingface processor
#     processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
#     hug_inputs = processor(images=image, return_tensors="pt")

#     #my processor
#     image_processor=my_ImageProcessorPil()
#     my_inputs=image_processor.process_images(
#         images =[image],
#         size = (224, 224),
#         resample =PILImageResampling.BICUBIC
#         )
    
#     #compare my_inputs & hug_inputs
#     print(f'my_inputs:{my_inputs}')
#     print(f'hug_inputs:{hug_inputs["pixel_values"]}')
#     bool_equal=torch.eq(my_inputs, hug_inputs["pixel_values"])
#     print(bool_equal)
#     print(torch.all(bool_equal))
#     # print the differences
#     diff = (my_inputs - hug_inputs["pixel_values"]).abs()
#     print(
#         f'max={diff.max().item():.8e}, '
#         f'mean={diff.mean().item():.8e}, '
#         f'allclose={torch.allclose(my_inputs, hug_inputs["pixel_values"], rtol=1e-5, atol=1e-6)}'
#     )
