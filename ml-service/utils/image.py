import base64
from io import BytesIO
from PIL import Image


def decode_image(data_url: str) -> Image.Image:
    if ',' in data_url:
        data_url = data_url.split(',', 1)[1]

    image = Image.open(BytesIO(base64.b64decode(data_url)))

    if image.mode == 'RGBA':
        bg = Image.new('RGB', image.size, (255, 255, 255))
        bg.paste(image, mask=image.split()[3])
        return bg

    return image.convert('RGB')
