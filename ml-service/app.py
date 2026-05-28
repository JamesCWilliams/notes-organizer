"""

"""

from flask import Flask, request, jsonify
from utils.image import decode_image
from inference.transcribe import transcribe as run_transcribe


app = Flask(__name__)


@app.route('/transcribe', methods=['POST'])
def transcribe():
    data = request.get_json()
    image = decode_image(data['image'])
    text = run_transcribe(image)
    return jsonify({'text': text})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
