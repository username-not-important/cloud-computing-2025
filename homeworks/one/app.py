from transformers import pipeline
import time, datetime, os
from flask import Flask, request, jsonify

sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def log_metrics(input_text, start_time, end_time, prediction):
    timestamp = datetime.datetime.now().isoformat()
    latency_ms = (end_time - start_time) * 1000
    log_file = os.environ.get('METRICS_LOG_FILE', 'inference_metrics.log')

    log_entry = f"Timestamp: {timestamp}, Input Text: '{input_text}', Inference Latency (ms): {latency_ms:.2f}, Prediction: {prediction}"
    with open(log_file, 'a') as f:
        f.write(log_entry + '\n')
    print(log_entry)

app = Flask(__name__)

@app.route("/infer", methods=["POST"])
def infer():
    input_text = request.json.get('text', '')
    start_time = time.time()
    prediction_result = sentiment_pipeline(input_text)[0]
    end_time = time.time()
    prediction_label = prediction_result['label']

    log_metrics(input_text, start_time, end_time, prediction_label)

    return jsonify({"input_text": input_text, "sentiment": prediction_label}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)