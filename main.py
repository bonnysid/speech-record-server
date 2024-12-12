from flask import Flask, request, jsonify
import whisper
from flask_cors import CORS, cross_origin

# Загрузка модели
model = whisper.load_model("base")

app = Flask(__name__)
cors = CORS(app)

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    filepath = f"uploads/{file.filename}"
    file.save(filepath)

    try:
        result = model.transcribe(filepath, language="ru")
        return jsonify({'transcript': result["text"]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
