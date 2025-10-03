import torch
from comfy.clip_vision import clip_preprocess, Output
import comfy.model_management as model_management

try:
    import torchvision.transforms.v2 as T
except ImportError:
    import torchvision.transforms as T


def split_tiles(embeds, num_split):
    _, H, W, _ = embeds.shape
    out = []
    for x in embeds:
        x = x.unsqueeze(0)
        h, w = H // num_split, W // num_split
        x_split = torch.cat(
            [x[:, i * h:(i + 1) * h, j * w:(j + 1) * w, :] for i in range(num_split) for j in range(num_split)], dim=0)
        out.append(x_split)

    x_split = torch.stack(out, dim=0)

    return x_split


def merge_hiddenstates(x, tiles):
    chunk_size = tiles * tiles
    x = x.split(chunk_size)

    out = []
    for embeds in x:
        num_tiles = embeds.shape[0]
        tile_size = int((embeds.shape[1] - 1) ** 0.5)
        grid_size = int(num_tiles ** 0.5)

        # Extract class tokens
        class_tokens = embeds[:, 0, :]  # Save class tokens: [num_tiles, embeds[-1]]
        avg_class_token = class_tokens.mean(dim=0, keepdim=True).unsqueeze(
            0)  # Average token, shape: [1, 1, embeds[-1]]

        patch_embeds = embeds[:, 1:, :]  # Shape: [num_tiles, tile_size^2, embeds[-1]]
        reshaped = patch_embeds.reshape(grid_size, grid_size, tile_size, tile_size, embeds.shape[-1])

        merged = torch.cat([torch.cat([reshaped[i, j] for j in range(grid_size)], dim=1)
                            for i in range(grid_size)], dim=0)

        merged = merged.unsqueeze(0)  # Shape: [1, grid_size*tile_size, grid_size*tile_size, embeds[-1]]

        # Pool to original size
        pooled = torch.nn.functional.adaptive_avg_pool2d(merged.permute(0, 3, 1, 2), (tile_size, tile_size)).permute(0,
                                                                                                                     2,
                                                                                                                     3,
                                                                                                                     1)
        flattened = pooled.reshape(1, tile_size * tile_size, embeds.shape[-1])

        # Add back the class token
        with_class = torch.cat([avg_class_token, flattened], dim=1)  # Shape: original shape
        out.append(with_class)

    out = torch.cat(out, dim=0)

    return out


def merge_embeddings(x, tiles):  # TODO: this needs so much testing that I don't even
    chunk_size = tiles * tiles
    x = x.split(chunk_size)

    out = []
    for embeds in x:
        num_tiles = embeds.shape[0]
        grid_size = int(num_tiles ** 0.5)
        tile_size = int(embeds.shape[1] ** 0.5)
        reshaped = embeds.reshape(grid_size, grid_size, tile_size, tile_size)

        # Merge the tiles
        merged = torch.cat([torch.cat([reshaped[i, j] for j in range(grid_size)], dim=1)
                            for i in range(grid_size)], dim=0)

        merged = merged.unsqueeze(0)  # Shape: [1, grid_size*tile_size, grid_size*tile_size]

        # Pool to original size
        pooled = torch.nn.functional.adaptive_avg_pool2d(merged,
                                                         (tile_size, tile_size))  # pool to [1, tile_size, tile_size]
        pooled = pooled.flatten(1)  # flatten to [1, tile_size^2]
        out.append(pooled)
    out = torch.cat(out, dim=0)

    return out


def encode_image_masked(clip_vision, image, mask=None, batch_size=0, tiles=1, ratio=1.0, clipvision_size=224):
    # full image embeds
    embeds = encode_image_masked_(clip_vision, image, mask, batch_size, clipvision_size=clipvision_size)
    tiles = min(tiles, 16)

    if tiles > 1:
        # split in tiles
        image_split = split_tiles(image, tiles)

        # get the embeds for each tile
        embeds_split = Output()
        for i in image_split:
            encoded = encode_image_masked_(clip_vision, i, mask, batch_size, clipvision_size=clipvision_size)
            if not hasattr(embeds_split, "image_embeds"):
                # embeds_split["last_hidden_state"] = encoded["last_hidden_state"]
                embeds_split["image_embeds"] = encoded["image_embeds"]
                embeds_split["penultimate_hidden_states"] = encoded["penultimate_hidden_states"]
            else:
                # embeds_split["last_hidden_state"] = torch.cat((embeds_split["last_hidden_state"], encoded["last_hidden_state"]), dim=0)
                embeds_split["image_embeds"] = torch.cat((embeds_split["image_embeds"], encoded["image_embeds"]), dim=0)
                embeds_split["penultimate_hidden_states"] = torch.cat(
                    (embeds_split["penultimate_hidden_states"], encoded["penultimate_hidden_states"]), dim=0)

        # embeds_split['last_hidden_state'] = merge_hiddenstates(embeds_split['last_hidden_state'])
        embeds_split["image_embeds"] = merge_embeddings(embeds_split["image_embeds"], tiles)
        embeds_split["penultimate_hidden_states"] = merge_hiddenstates(embeds_split["penultimate_hidden_states"], tiles)

        # embeds['last_hidden_state'] = torch.cat([embeds_split['last_hidden_state'], embeds['last_hidden_state']])
        if embeds['image_embeds'].shape[
            0] > 1:  # if we have more than one image we need to average the embeddings for consistency
            embeds['image_embeds'] = embeds['image_embeds'] * ratio + embeds_split['image_embeds'] * (1 - ratio)
            embeds['penultimate_hidden_states'] = embeds['penultimate_hidden_states'] * ratio + embeds_split[
                'penultimate_hidden_states'] * (1 - ratio)
            # embeds['image_embeds'] = (embeds['image_embeds']*ratio + embeds_split['image_embeds']) / 2
            # embeds['penultimate_hidden_states'] = (embeds['penultimate_hidden_states']*ratio + embeds_split['penultimate_hidden_states']) / 2
        else:  # otherwise we can concatenate them, they can be averaged later
            embeds['image_embeds'] = torch.cat([embeds['image_embeds'] * ratio, embeds_split['image_embeds']])
            embeds['penultimate_hidden_states'] = torch.cat(
                [embeds['penultimate_hidden_states'] * ratio, embeds_split['penultimate_hidden_states']])

    # del embeds_split

    return embeds


def encode_image_masked_(clip_vision, images, mask=None, batch_size=0, clipvision_size=224):
    model_management.load_model_gpu(clip_vision.patcher)
    outputs = Output()

    if batch_size == 0:
        batch_size = images.shape[0]
    elif batch_size > images.shape[0]:
        batch_size = images.shape[0]

    image_batch = torch.split(images, batch_size, dim=0)

    for img in image_batch:
        img = img.to(clip_vision.load_device)
        pixel_values = clip_preprocess(img, size=clipvision_size).float()

        # TODO: support for multiple masks
        if mask is not None:
            pixel_values = pixel_values * mask.to(clip_vision.load_device)

        out = clip_vision.model(pixel_values=pixel_values, intermediate_output=-2)
        if not hasattr(outputs, "last_hidden_state"):
            outputs["last_hidden_state"] = out[0].to(model_management.intermediate_device())
            outputs["image_embeds"] = out[2].to(model_management.intermediate_device())
            outputs["penultimate_hidden_states"] = out[1].to(model_management.intermediate_device())
        else:
            outputs["last_hidden_state"] = torch.cat(
                (outputs["last_hidden_state"], out[0].to(model_management.intermediate_device())), dim=0)
            outputs["image_embeds"] = torch.cat(
                (outputs["image_embeds"], out[2].to(model_management.intermediate_device())), dim=0)
            outputs["penultimate_hidden_states"] = torch.cat(
                (outputs["penultimate_hidden_states"], out[1].to(model_management.intermediate_device())), dim=0)

    del img, pixel_values, out
    torch.cuda.empty_cache()

    return outputs


def tensor_to_size(source, dest_size):
    if isinstance(dest_size, torch.Tensor):
        dest_size = dest_size.shape[0]
    source_size = source.shape[0]

    if source_size < dest_size:
        shape = [dest_size - source_size] + [1] * (source.dim() - 1)
        source = torch.cat((source, source[-1:].repeat(shape)), dim=0)
    elif source_size > dest_size:
        source = source[:dest_size]

    return source


def min_(tensor_list):
    # return the element-wise min of the tensor list.
    x = torch.stack(tensor_list)
    mn = x.min(axis=0)[0]
    return torch.clamp(mn, min=0)


def max_(tensor_list):
    # return the element-wise max of the tensor list.
    x = torch.stack(tensor_list)
    mx = x.max(axis=0)[0]
    return torch.clamp(mx, max=1)


# From https://github.com/Jamy-L/Pytorch-Contrast-Adaptive-Sharpening/
def contrast_adaptive_sharpening(image, amount):
    img = T.functional.pad(image, (1, 1, 1, 1)).cpu()

    a = img[..., :-2, :-2]
    b = img[..., :-2, 1:-1]
    c = img[..., :-2, 2:]
    d = img[..., 1:-1, :-2]
    e = img[..., 1:-1, 1:-1]
    f = img[..., 1:-1, 2:]
    g = img[..., 2:, :-2]
    h = img[..., 2:, 1:-1]
    i = img[..., 2:, 2:]

    # Computing contrast
    cross = (b, d, e, f, h)
    mn = min_(cross)
    mx = max_(cross)

    diag = (a, c, g, i)
    mn2 = min_(diag)
    mx2 = max_(diag)
    mx = mx + mx2
    mn = mn + mn2

    # Computing local weight
    inv_mx = torch.reciprocal(mx)
    amp = inv_mx * torch.minimum(mn, (2 - mx))

    # scaling
    amp = torch.sqrt(amp)
    w = - amp * (amount * (1 / 5 - 1 / 8) + 1 / 8)
    div = torch.reciprocal(1 + 4 * w)

    output = ((b + d + f + h) * w + e) * div
    output = torch.nan_to_num(output)
    output = output.clamp(0, 1)

    return output


def tensor_to_image(tensor):
    image = tensor.mul(255).clamp(0, 255).byte().cpu()
    image = image[..., [2, 1, 0]].numpy()
    return image


def image_to_tensor(image):
    tensor = torch.clamp(torch.from_numpy(image).float() / 255., 0, 1)
    tensor = tensor[..., [2, 1, 0]]
    return tensor