import numpy as np
from PIL import Image

def resize_with_padding(img, target_size):
    target_width, target_height = target_size
    original_width, original_height = img.size

    ratio = min(target_width / original_width, target_height / original_height)
    new_size = (int(original_width * ratio), int(original_height * ratio))
    resized = img.resize(new_size, Image.Resampling.LANCZOS)

    padded_img = Image.new('RGB', (target_width, target_height), (255, 255, 255))
    paste_position = ((target_width - new_size[0]) // 2, (target_height - new_size[1]) // 2)
    padded_img.paste(resized, paste_position)

    return padded_img


def preprocess_image(image, target_size=(80,35)):
    image = resize_with_padding(image, target_size)
    return np.expand_dims(image, axis=0)