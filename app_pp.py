from flask import Flask, render_template_string, request, send_file
import os
import editdistance
import re
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

telugu_pattern = re.compile(r'^[\u0C00-\u0C7F]+$')

def is_telugu_word(word):
    return bool(telugu_pattern.fullmatch(word))

def read_data(file_path):
    gt_list, pred_list, prob_list = [], [], []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 3:
                continue
            gt, pred, prob = parts
            gt_list.append(gt)
            pred_list.append(pred)
            prob_list.append(float(prob))
    return gt_list, pred_list, prob_list

def read_dictionary(dict_path):
    with open(dict_path, 'r', encoding='utf-8') as f:
        return set(word.strip() for word in f if is_telugu_word(word.strip()))

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template_string('''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <title>Telugu OCR Post-Processing</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        </head>
        <body class="bg-light">
            <div class="container py-5">
                <h2 class="mb-4 text-center text-primary">📝 Telugu OCR Post-Processing Tool</h2>
                <form method="POST" action="/review" enctype="multipart/form-data" class="bg-white p-4 rounded shadow">
                    <div class="mb-3">
                        <label class="form-label">📂 Input File (.txt)</label>
                        <input type="file" name="input_file" class="form-control" required>
                    </div>
                    <div class="mb-3">
                        <label class="form-label">📖 Dictionary File (.txt)</label>
                        <input type="file" name="dict_file" class="form-control" required>
                    </div>
                    <div class="mb-3">
                        <label class="form-label">🎯 Probability Threshold</label>
                        <input type="text" name="prob_threshold" class="form-control" value="0.90">
                    </div>
                    <div class="mb-3">
                        <label class="form-label">✂️ Edit Distance Threshold</label>
                        <input type="text" name="edit_dist_threshold" class="form-control" value="2">
                    </div>
                    <button type="submit" class="btn btn-primary w-100">Review and Correct ✏️</button>
                </form>
            </div>
        </body>
        </html>
    ''')

@app.route('/review', methods=['POST'])
def review():
    input_file = request.files['input_file']
    dict_file = request.files['dict_file']
    prob_threshold = float(request.form.get('prob_threshold', 0.9))
    dist_threshold = int(request.form.get('edit_dist_threshold', 2))

    input_path = os.path.join(UPLOAD_FOLDER, secure_filename(input_file.filename))
    dict_path = os.path.join(UPLOAD_FOLDER, secure_filename(dict_file.filename))
    input_file.save(input_path)
    dict_file.save(dict_path)

    gt_list, pred_list, prob_list = read_data(input_path)
    dictionary = read_dictionary(dict_path)

    review_data = []
    for idx, (pred, prob) in enumerate(zip(pred_list, prob_list)):
        if not is_telugu_word(pred) or pred in dictionary or prob > prob_threshold:
            continue
        first_char = pred[0]
        candidates = [w for w in dictionary if w.startswith(first_char)]
        distances = [(w, editdistance.eval(pred, w)) for w in candidates]
        if not distances:
            continue
        min_dist = min(d[1] for d in distances)
        if min_dist > dist_threshold:
            continue
        tied = [w for w, d in distances if d == min_dist]
        if len(tied) > 1:
            review_data.append({'index': idx, 'word': pred, 'prob': prob, 'candidates': tied})

    return render_template_string('''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <title>Review Corrections</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        </head>
        <body class="bg-light">
            <div class="container py-5">
                <h3 class="mb-4 text-success text-center">🔍 Review and Select Corrections</h3>
                <form method="POST" action="/process" class="bg-white p-4 rounded shadow">
                    {% for item in review_data %}
                        <div class="mb-3">
                            <label class="form-label"><strong>{{ item.word }}</strong> (Confidence: {{ item.prob }})</label>
                            <select name="correction_{{ item.index }}" class="form-select">
                                {% for cand in item.candidates %}
                                    <option value="{{ cand }}">{{ cand }}</option>
                                {% endfor %}
                            </select>
                        </div>
                    {% endfor %}
                    {% for key, value in request.form.items() %}
                        <input type="hidden" name="{{ key }}" value="{{ value }}">
                    {% endfor %}
                    <input type="hidden" name="input_path" value="{{ input_path }}">
                    <input type="hidden" name="dict_path" value="{{ dict_path }}">
                    <button type="submit" class="btn btn-success w-100">✅ Apply Corrections & Download</button>
                </form>
            </div>
        </body>
        </html>
    ''', review_data=review_data, input_path=input_path, dict_path=dict_path)

@app.route('/process', methods=['POST'])
def process():
    input_path = request.form['input_path']
    gt_list, pred_list, prob_list = read_data(input_path)

    for key in request.form:
        if key.startswith("correction_"):
            idx = int(key.split("_")[1])
            pred_list[idx] = request.form[key]

    def word_accuracy(gt, pred): return sum(g == p for g, p in zip(gt, pred)) / len(gt)
    def char_accuracy(gt, pred): return sum(len(g) - editdistance.eval(g, p) for g, p in zip(gt, pred)) / sum(len(g) for g in gt)

    wrr = word_accuracy(gt_list, pred_list)
    crr = char_accuracy(gt_list, pred_list)

    output_path = os.path.join(OUTPUT_FOLDER, "corrected_output.txt")
    with open(output_path, 'w', encoding='utf-8') as f:
        for gt, pred, prob in zip(gt_list, pred_list, prob_list):
            f.write(f"{gt}\t{pred}\t{prob:.4f}\n")
        f.write(f"\nWRR: {wrr:.4f}\nCRR: {crr:.4f}\n")

    return send_file(output_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
