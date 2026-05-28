import base64
from io import BytesIO

from PIL import Image

from utils.image import decode_image


def _make_data_url(mode: str) -> str:
    img = Image.new(mode, (64, 64), color=(0, 0, 0, 255) if mode == 'RGBA' else (0, 0, 0))
    buf = BytesIO()
    img.save(buf, format='PNG')
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f'data:image/png;base64,{b64}'


def test_decode_strips_data_url_prefix():
    result = decode_image(_make_data_url('RGB'))
    assert isinstance(result, Image.Image)


def test_decode_rgba_composites_to_white_background():
    result = decode_image(_make_data_url('RGBA'))
    assert result.mode == 'RGB'


def test_decode_rgb_stays_rgb():
    result = decode_image(_make_data_url('RGB'))
    assert result.mode == 'RGB'


def test_decode_without_data_url_prefix():
    img = Image.new('RGB', (64, 64))
    buf = BytesIO()
    img.save(buf, format='PNG')
    b64 = base64.b64encode(buf.getvalue()).decode()
    result = decode_image(b64)
    assert result.mode == 'RGB'
