import os
import torch
from PIL import Image, ImageOps
import numpy as np
import folder_paths

from .utils import encode_image_masked, contrast_adaptive_sharpening

try:
    import torchvision.transforms.v2 as T
except ImportError:
    import torchvision.transforms as T

class PrepClipVisionBatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images_folder": ("STRING", {"default": r""}),
                "interpolation": (["LANCZOS", "BICUBIC", "HAMMING", "BILINEAR", "BOX", "NEAREST"],),
                "crop_position": (["top", "bottom", "left", "right", "center", "pad"],),
                "sharpening": ("FLOAT", {"default": 0.0, "min": 0, "max": 1, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "preprocess_image_batch"
    OUTPUT_NODE = True
    CATEGORY = "ipadapter"

    def preprocess_single_image(self, image, interpolation="LANCZOS", crop_position="center", sharpening=0.0):
        size = (224, 224)
        _, oh, ow, _ = image.shape
        output = image.permute([0, 3, 1, 2])

        if crop_position == "pad":
            if oh != ow:
                if oh > ow:
                    pad = (oh - ow) // 2
                    pad = (pad, 0, pad, 0)
                elif ow > oh:
                    pad = (ow - oh) // 2
                    pad = (0, pad, 0, pad)
                output = T.functional.pad(output, pad, fill=0)
        else:
            crop_size = min(oh, ow)
            x = (ow - crop_size) // 2
            y = (oh - crop_size) // 2
            if "top" in crop_position:
                y = 0
            elif "bottom" in crop_position:
                y = oh - crop_size
            elif "left" in crop_position:
                x = 0
            elif "right" in crop_position:
                x = ow - crop_size

            x2 = x + crop_size
            y2 = y + crop_size

            output = output[:, :, y:y2, x:x2]

        imgs = []
        for img in output:
            img = T.ToPILImage()(img)
            img = img.resize(size, resample=Image.Resampling[interpolation])
            imgs.append(T.ToTensor()(img))
        output = torch.stack(imgs, dim=0)
        del imgs, img

        if sharpening > 0:
            output = contrast_adaptive_sharpening(output, sharpening)

        output = output.permute([0, 2, 3, 1])

        return (output,)

    VALID_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}

    def path_to_image_tensor(self, path: str) -> torch.Tensor:
        """Return (1, H, W, 3) float32 [0..1], EXIF-corrected, RGB."""
        im = Image.open(path)
        im = ImageOps.exif_transpose(im)
        if im.mode != "RGB":
            im = im.convert("RGB")
        arr = np.asarray(im, dtype=np.float32) / 255.0  # (H, W, 3)
        t = torch.from_numpy(arr).unsqueeze(0).contiguous()  # (1, H, W, 3)
        return t

    def preprocess_image_batch(self, images_folder, interpolation, crop_position, sharpening):
        tensors = []
        for name in sorted(os.listdir(images_folder)):
            ext = os.path.splitext(name)[1].lower()
            if ext not in self.VALID_EXTS:
                continue
            path = os.path.join(images_folder, name)

            img = self.path_to_image_tensor(path)  # (1,H,W,3) in [0,1]
            (prepped,) = self.preprocess_single_image(
                img, interpolation, crop_position, sharpening
            )
            tensors.append(prepped)

        if not tensors:
            raise RuntimeError("No valid images found in folder.")

        batch = torch.cat(tensors, dim=0)  # (B,224,224,3)
        return (batch,)

class IPAEncoderBatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "ipadapter": ("IPADAPTER",),
            "images": ("IMAGE",),
            "weight": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 3.0, "step": 0.01}),
            "method": (["concat", "add", "subtract", "average", "norm average", "max", "min"], ),
            "batch_size": ("INT", {"default": 4, "min": 1, "max": 4096, "step": 1}),
        },
            "optional": {
                "mask": ("MASK",),
                "clip_vision": ("CLIP_VISION",),
            }
        }

    RETURN_TYPES = ("EMBEDS", "EMBEDS",)
    RETURN_NAMES = ("pos_embed", "neg_embed",)
    FUNCTION = "encode_batch"
    CATEGORY = "hapless"

    def combine_embeds(self, embeds, method):
        embeds = [embed for embed in embeds if embed is not None]
        embeds = torch.cat(embeds, dim=0)

        if method == "add":
            embeds = torch.sum(embeds, dim=0).unsqueeze(0)
        elif method == "subtract":
            embeds = embeds[0] - torch.mean(embeds[1:], dim=0)
            embeds = embeds.unsqueeze(0)
        elif method == "average":
            embeds = torch.mean(embeds, dim=0).unsqueeze(0)
        elif method == "norm average":
            embeds = torch.mean(embeds / torch.norm(embeds, dim=0, keepdim=True), dim=0).unsqueeze(0)
        elif method == "max":
            embeds = torch.max(embeds, dim=0).values.unsqueeze(0)
        elif method == "min":
            embeds = torch.min(embeds, dim=0).values.unsqueeze(0)

        return embeds

    def encode_batch(self, ipadapter, images, weight, batch_size, mask=None, clip_vision=None, method="concat"):
        if 'ipadapter' in ipadapter:
            ipadapter_model = ipadapter['ipadapter']['model']
            clip_vision = clip_vision if clip_vision is not None else ipadapter['clipvision']['model']
        else:
            ipadapter_model = ipadapter
            clip_vision = clip_vision

        if clip_vision is None:
            raise Exception("Missing CLIPVision model.")

        is_plus = "proj.3.weight" in ipadapter_model["image_proj"] or "latents" in ipadapter_model[
            "image_proj"] or "perceiver_resampler.proj_in.weight" in ipadapter_model["image_proj"]
        is_kwai_kolors = is_plus and "layers.0.0.to_out.weight" in ipadapter_model["image_proj"] and \
                         ipadapter_model["image_proj"]["layers.0.0.to_out.weight"].shape[0] == 2048

        clipvision_size = 224 if not is_kwai_kolors else 336

        # resize and crop the mask to 224x224
        if mask is not None and mask.shape[1:3] != torch.Size([clipvision_size, clipvision_size]):
            mask = mask.unsqueeze(1)
            transforms = T.Compose([
                T.CenterCrop(min(mask.shape[2], mask.shape[3])),
                T.Resize((clipvision_size, clipvision_size), interpolation=T.InterpolationMode.BICUBIC, antialias=True),
            ])
            mask = transforms(mask).squeeze(1)
            # mask = T.Resize((image.shape[1], image.shape[2]), interpolation=T.InterpolationMode.BICUBIC, antialias=True)(mask.unsqueeze(1)).squeeze(1)
        embeds = encode_image_masked(
            clip_vision,
            images,
            mask,
            batch_size=batch_size,
            clipvision_size=clipvision_size,
        )

        if is_plus:
            img_cond_embeds = embeds.penultimate_hidden_states
            # Make an unconditional batch matching B
            B = images.shape[0]
            zeros = torch.zeros([B, clipvision_size, clipvision_size, 3])
            uncond = encode_image_masked(
                clip_vision, zeros, None, batch_size=batch_size, clipvision_size=clipvision_size
            )
            img_uncond_embeds = uncond.penultimate_hidden_states
        else:
            img_cond_embeds = embeds.image_embeds
            img_uncond_embeds = torch.zeros_like(img_cond_embeds)

        if weight != 1:
            img_cond_embeds = img_cond_embeds * weight
        # potentially better for plus models which I don't know what they are
        # img_cond_embeds = img_uncond_embeds + weight * (img_cond_embeds - img_uncond_embeds)

        pos = self.combine_embeds([img_cond_embeds], method)
        neg = self.combine_embeds([img_uncond_embeds], method)

        return (pos, neg,)

class IPALoadEmbeds:
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = os.path.join(folder_paths.models_dir, "ipadapter", "embeds")
        files = [os.path.relpath(os.path.join(root, file), input_dir) for root, dirs, files in os.walk(input_dir) for file in files if file.endswith('.ipadpt')]
        return {"required": {"embeds": [sorted(files), ]}, }

    RETURN_TYPES = ("EMBEDS", )
    FUNCTION = "load"
    CATEGORY = "ipadapter"

    def load(self, embeds):
        path = os.path.join(folder_paths.models_dir, "ipadapter", "embeds", embeds)
        return (torch.load(path).cpu(), )

class IPASaveEmbeds:
    def __init__(self):
        self.output_dir = os.path.join(folder_paths.models_dir, "ipadapter", "embeds")

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "embeds": ("EMBEDS",),
            "filename_prefix": ("STRING", {"default": "IP_embeds"})
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "ipadapter"

    def save(self, embeds, filename_prefix):
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir)
        file = f"{filename}_{counter:05}.ipadpt"
        file = os.path.join(full_output_folder, file)

        torch.save(embeds, file)
        return (None, )