from flask import Flask, request, jsonify
from flask_cors import CORS
from utils.image import decode_image, ImageDecodeError
from inference.transcribe import transcribe as run_transcribe


app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB request cap
CORS(app, origins=[
    r'http://localhost(:\d+)?',   # dev server (e.g. :1420) and plain localhost
    r'http://127\.0\.0\.1(:\d+)?',
    'http://tauri.localhost',     # tauri v2 linux production
    'tauri://localhost',          # tauri v2 macos/windows production
])


@app.route('/transcribe', methods=['POST'])
def transcribe():
    data = request.get_json(silent=True)
    if not data or 'image' not in data:
        return jsonify({'error': 'Missing image field'}), 400
    if not isinstance(data['image'], str):
        return jsonify({'error': 'image must be a string'}), 400
    if 'strokes' not in data:
        return jsonify({'error': 'no strokes data included in request'}), 400

    try:
        image = decode_image(data['image'])
    except ImageDecodeError as e:
        return jsonify({'error': str(e)}), 400
    try:
        strokes = data['strokes']
    except Exception as e:
        return jsonify({'error': str(e)}), 400

    text = run_transcribe(image, strokes)
    return jsonify({'text': text})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
