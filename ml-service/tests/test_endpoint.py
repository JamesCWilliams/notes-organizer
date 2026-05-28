import base64
from io import BytesIO

import pytest
from PIL import Image

from app import app


@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def _png_data_url() -> str:
    img = Image.new('RGB', (64, 64), color=(255, 255, 255))
    buf = BytesIO()
    img.save(buf, format='PNG')
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f'data:image/png;base64,{b64}'


def test_transcribe_returns_text_key(client):
    response = client.post('/transcribe', json={'image': _png_data_url()})
    assert response.status_code == 200
    assert 'text' in response.get_json()


def test_transcribe_text_is_string(client):
    response = client.post('/transcribe', json={'image': _png_data_url()})
    assert isinstance(response.get_json()['text'], str)
