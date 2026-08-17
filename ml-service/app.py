import os

from flask import Flask, jsonify, request
from flask_cors import CORS

from embedding.image import embed_image
from embedding.strokes import embed_strokes
from embedding.text import embed_text
from inference.transcribe import transcribe as run_transcribe
from utils.image import ImageDecodeError, decode_image
from utils.rows import StrokeDataError

ML_DEBUG = os.environ.get('ML_DEBUG', 'false').lower() == 'true'


app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB request cap
CORS(app, origins=[
    r'http://localhost(:\d+)?',   # dev server (e.g. :1420) and plain localhost
    r'http://127\.0\.0\.1(:\d+)?',
    'http://tauri.localhost',     # tauri v2 linux production
    'tauri://localhost',          # tauri v2 macos/windows production
])


def _parse_payload():
    """Validates the request body shared by /transcribe and /analyze.

    Returns (image, strokes, None) on success, or (None, None, response)
    where response is a ready-to-return Flask error response.
    """
    data = request.get_json(silent=True)
    if not data or 'image' not in data:
        return None, None, (jsonify({'error': 'Missing image field'}), 400)
    if not isinstance(data['image'], str):
        return None, None, (jsonify({'error': 'image must be a string'}), 400)
    if 'strokes' not in data:
        return None, None, (jsonify({'error': 'no strokes data included in request'}), 400)

    try:
        image = decode_image(data['image'])
    except ImageDecodeError as e:
        return None, None, (jsonify({'error': str(e)}), 400)

    return image, data['strokes'], None


@app.route('/transcribe', methods=['POST'])
def transcribe():
    image, strokes, error = _parse_payload()
    if error:
        return error
    if not image:
        return jsonify({'error': 'image is empty'}), 400
    if not strokes:
        return jsonify({'error': 'strokes list is empty'}), 400
    try:
        text = run_transcribe(image, strokes, ML_DEBUG)
    except StrokeDataError as e:
        return jsonify({'error': str(e)}), 400
    return jsonify({'text': text})


@app.route('/analyze', methods=['POST'])
def analyze():
    """Transcription plus embeddings, for enriching saved notes."""
    image, strokes, error = _parse_payload()
    if error:
        return error
    if not image:
        return jsonify({'error': 'image is empty'}), 400
    if not strokes:
        return jsonify({'error': 'strokes list is empty'}), 400
    try:
        text = run_transcribe(image, strokes, ML_DEBUG)
    except StrokeDataError as e:
        return jsonify({'error': str(e)}), 400

    return jsonify({
        'text': text,
        'embeddings': {
            # A note with no recognizable text (e.g. a pure doodle) gets no
            # text vector rather than a meaningless one.
            'text': embed_text(text) if text.strip() else None,
            'image': embed_image(image),
            # None when no stroke encoder has been exported yet.
            'strokes': embed_strokes(strokes),
        },
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
