import base64
import binascii
from io import BytesIO
from PIL import Image, UnidentifiedImageError

MAX_IMAGE_PIXELS = 4000 * 4000


class ImageDecodeError(ValueError):
    pass


def decode_image(data_url: str) -> Image.Image:
    if ',' in data_url:
        data_url = data_url.split(',', 1)[1]

    try:
        image_bytes = base64.b64decode(data_url, validate=True)
    except binascii.Error:
        raise ImageDecodeError('Invalid base64 data')

    try:
        image = Image.open(BytesIO(image_bytes))
        image.verify()
        image = Image.open(BytesIO(image_bytes))
    except (UnidentifiedImageError, Exception):
        raise ImageDecodeError('Could not decode image')

    if image.width * image.height > MAX_IMAGE_PIXELS:
        raise ImageDecodeError('Image exceeds maximum allowed dimensions')

    if image.mode == 'RGBA':
        bg = Image.new('RGB', image.size, (255, 255, 255))
        bg.paste(image, mask=image.split()[3])
        return bg

    return image.convert('RGB')
