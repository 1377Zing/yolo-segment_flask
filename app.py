from flask import render_template, send_from_directory, url_for, jsonify, Flask, request, send_file
from flask_cors import CORS
from yolo import YOLO
import io
import base64
import zipfile
import os
from fluorcent import *
app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER_SINGLE = os.path.join('static', 'uploads_single')
UPLOAD_FOLDER_MULTIPLE = os.path.join('static', 'uploads_multiple')
PROCESSED_FOLDER = os.path.join('static', 'processed')
os.makedirs(UPLOAD_FOLDER_SINGLE, exist_ok=True)
os.makedirs(UPLOAD_FOLDER_MULTIPLE, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
def process_folder(folder_path, info):
    color_results = {'red': [], 'green': [], 'blue': [], 'yellow': []}
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
            file_path = os.path.join(folder_path, filename)
            processed_image, fluorescence_data = process_image(file_path, info)
            if processed_image:
                processed_image.save(os.path.join(PROCESSED_FOLDER, f"processed_{filename}"))
                for color, identifier, total_fluorescence, avg_fluorescence in fluorescence_data:
                    color_results[color].append((identifier, total_fluorescence, avg_fluorescence))
    return color_results
def process_folder2(folder_path, info, selectedNumber):
    color_results = {}
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(
                ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
            file_path = os.path.join(folder_path, filename)
            processed_image, fluorescence_data = process_tail_image(file_path, info, selectedNumber)

            if processed_image:
                try:
                    processed_image.save(os.path.join(PROCESSED_FOLDER, f"processed_{filename}"))
                except Exception as e:
                    print(f"Error saving image: {e}")

                for res in fluorescence_data:

                    color = res["color"]
                    head_total_fluorescence = res["head"]["total_fluorescence"]
                    head_avg_fluorescence = res["head"]["avg_fluorescence"]
                    tail_total_fluorescence = res["tail"]["total_fluorescence"]
                    tail_avg_fluorescence = res["tail"]["avg_fluorescence"]

                    if color not in color_results:
                        color_results[color] = []

                    color_results[color].append({
                        "id": res['id'],
                        "head": [head_total_fluorescence, head_avg_fluorescence],
                        "tail": [tail_total_fluorescence, tail_avg_fluorescence]
                    })

    return color_results

def clear_folder(folder_path):
    try:
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                for root, dirs, files in os.walk(file_path, topdown=False):
                    for name in files:
                        os.remove(os.path.join(root, name))
                    for name in dirs:
                        os.rmdir(os.path.join(root, name))
                os.rmdir(file_path)

    except Exception as e:
        print(f"Error clearing folder: {e}")

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/upload', methods=['POST'])
def upload_files():
    uploaded_files = []

    if 'single_file' in request.files:
        clear_folder(UPLOAD_FOLDER_SINGLE)
        single_file = request.files['single_file']
        if single_file:
            filename, file_ext = os.path.splitext(single_file.filename)
            file_ext = file_ext.lower()

            if file_ext == '.tif':
                img = Image.open(single_file)
                new_filename = filename + '.png'
                save_path = os.path.join(UPLOAD_FOLDER_SINGLE, new_filename)
                img.save(save_path, format='PNG')
                uploaded_files.append(url_for('uploaded_file_single', filename=new_filename))
            else:
                save_path = os.path.join(UPLOAD_FOLDER_SINGLE, single_file.filename)
                single_file.save(save_path)
                uploaded_files.append(url_for('uploaded_file_single', filename=single_file.filename))

    if 'multiple_files' in request.files:
        clear_folder(UPLOAD_FOLDER_MULTIPLE)
        multiple_files = request.files.getlist('multiple_files')
        for file in multiple_files:
            if file:

                filename, file_ext = os.path.splitext(file.filename)
                file_ext = file_ext.lower()
                if file_ext == '.tif':
                    img = Image.open(file)
                    new_filename = filename + '.png'
                    save_path = os.path.join(UPLOAD_FOLDER_MULTIPLE, new_filename)
                    img.save(save_path, format='PNG')
                    uploaded_files.append(url_for('uploaded_file_multiple', filename=new_filename))
                else:

                    save_path = os.path.join(UPLOAD_FOLDER_MULTIPLE, file.filename)
                    file.save(save_path)
                    uploaded_files.append(url_for('uploaded_file_multiple', filename=file.filename))

    return jsonify({"uploaded_files": uploaded_files})

@app.route('/process', methods=['POST'])
def mainProcess():
    clear_folder(PROCESSED_FOLDER)
    yolo = YOLO()
    results = {'red': [], 'green': [], 'blue': [], 'yellow': []}
    info = None

    uploaded_files_single = os.listdir(UPLOAD_FOLDER_SINGLE)
    for filename in uploaded_files_single:
        file_path = os.path.join(UPLOAD_FOLDER_SINGLE, filename)
        try:
            image = Image.open(file_path)
        except:
            continue
        crop = False
        count = True
        r_image, info = yolo.detect_image(image, crop=crop, count=count)
        processed_image_path = os.path.join(PROCESSED_FOLDER, f"processed_{filename}")
        r_image.save(processed_image_path)

    uploaded_files_multiple = os.listdir(UPLOAD_FOLDER_MULTIPLE)
    color_results = process_folder(UPLOAD_FOLDER_MULTIPLE, info)
    for color, data in color_results.items():
        results[color].extend(data)

    processed_images = [f"processed_{filename}" for filename in uploaded_files_single + uploaded_files_multiple]

    return jsonify({
        'processed_images': processed_images,
        'results': results
    })


@app.route('/process2', methods=['POST'])
def mainProcess2():
    data = request.get_json()
    selectedNumber = data.get('selectedNumber')
    if selectedNumber is None:
        return jsonify({"error": "Missing selectedNumber parameter"}), 400
    clear_folder(PROCESSED_FOLDER)
    yolo = YOLO()
    results = {'red': [], 'green': [], 'blue': [], 'yellow':[]}
    info = None

    uploaded_files_single = os.listdir(UPLOAD_FOLDER_SINGLE)
    for filename in uploaded_files_single:
        file_path = os.path.join(UPLOAD_FOLDER_SINGLE, filename)
        try:
            image = Image.open(file_path)
        except:
            continue
        crop = False
        count = True
        r_image, info = yolo.detect_image(image, crop=crop, count=count)
        processed_image_path = os.path.join(PROCESSED_FOLDER, f"processed_{filename}")
        r_image.save(processed_image_path)

    uploaded_files_multiple = os.listdir(UPLOAD_FOLDER_MULTIPLE)
    color_results = process_folder2(UPLOAD_FOLDER_MULTIPLE, info, selectedNumber)
    for color, data in color_results.items():
        results[color].extend(data)

    processed_images = [f"processed_{filename}" for filename in uploaded_files_single + uploaded_files_multiple]

    return jsonify({
        'processed_images': processed_images,
        'results': results
    })


@app.route('/uploads_single/<filename>')
def uploaded_file_single(filename):
    return send_from_directory(UPLOAD_FOLDER_SINGLE, filename)

@app.route('/uploads_multiple/<filename>')
def uploaded_file_multiple(filename):
    return send_from_directory(UPLOAD_FOLDER_MULTIPLE, filename)

@app.route('/processed/<filename>')
def processed_file(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)

@app.route('/download', methods=['POST'])
def download():
    data = request.get_json()
    excel_data = data.get('excelData')


    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zipf:

        excel_bytes = base64.b64decode(excel_data)
        zipf.writestr('result.xlsx', excel_bytes)

        folder_path = os.path.join(os.getcwd(), './static/processed')
        if os.path.exists(folder_path):
            for root, _, files in os.walk(folder_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    with open(file_path, 'rb') as f:
                        file_content = f.read()
                        relative_path = os.path.relpath(file_path, folder_path)
                        zipf.writestr(relative_path, file_content)

    zip_buffer.seek(0)

    return send_file(
        zip_buffer,
        as_attachment=True,
        download_name='result.zip',
        mimetype='application/zip'
    )

if __name__ == '__main__':
    app.run(debug=True)
