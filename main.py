from flask import Flask, request, jsonify
import whisper
import jiwer
from flask_cors import CORS
import os

# Загрузка модели
model = whisper.load_model("base")

app = Flask(__name__)
cors = CORS(app)

# Создание папки для загрузки файлов, если её нет
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    if 'files' not in request.files:
        return jsonify({'error': 'No files uploaded'}), 400

    files = request.files.getlist('files')
    reference_texts = request.form.getlist('referenceTexts')

    if len(files) != len(reference_texts):
        return jsonify({'error': 'Mismatch between files and reference texts'}), 400

    results = []

    for file, reference_text in zip(files, reference_texts):
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        try:
            # Распознавание речи
            result = model.transcribe(filepath, language="ru")
            transcript = result["text"]

            # Вычисление метрик
            wer = jiwer.wer(reference_text, transcript)
            cer = jiwer.cer(reference_text, transcript)

            # Добавление результата
            results.append({
                'fileName': file.filename,
                'transcript': transcript,
                'wer': wer,
                'cer': cer
            })

        except Exception as e:
            results.append({
                'fileName': file.filename,
                'error': str(e)
            })

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
