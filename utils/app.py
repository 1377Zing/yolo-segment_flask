from yolo import YOLO
from PIL import  ImageDraw, ImageFont
import os
from flask import Flask, render_template, request, send_from_directory, url_for, jsonify

app = Flask(__name__)

# 定义路径到 static 目录下的子文件夹
UPLOAD_FOLDER_SINGLE = os.path.join('static', 'uploads_single')
UPLOAD_FOLDER_MULTIPLE = os.path.join('static', 'uploads_multiple')
PROCESSED_FOLDER = os.path.join('static', 'processed')

# 创建这些文件夹，如果不存在的话
os.makedirs(UPLOAD_FOLDER_SINGLE, exist_ok=True)
os.makedirs(UPLOAD_FOLDER_MULTIPLE, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)


def clear_folder(folder_path):
    try:
        # 遍历文件夹中的所有文件和文件夹
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            # 删除文件
            if os.path.isfile(file_path):
                os.remove(file_path)
            # 删除文件夹及其内容
            elif os.path.isdir(file_path):
                # 可以选择是否递归删除子文件夹
                for root, dirs, files in os.walk(file_path, topdown=False):
                    for name in files:
                        os.remove(os.path.join(root, name))
                    for name in dirs:
                        os.rmdir(os.path.join(root, name))
                os.rmdir(file_path)
        print(f"Folder '{folder_path}' has been cleared.")
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
            # 获取文件名和扩展名
            filename, file_ext = os.path.splitext(single_file.filename)
            file_ext = file_ext.lower()

            if file_ext == '.tif':
                # 如果是 .tif 文件，转换为 .png
                img = Image.open(single_file)
                new_filename = filename + '.png'
                save_path = os.path.join(UPLOAD_FOLDER_SINGLE, new_filename)
                img.save(save_path, format='PNG')
                uploaded_files.append(url_for('uploaded_file_single', filename=new_filename))
            else:
                # 否则按原格式保存
                save_path = os.path.join(UPLOAD_FOLDER_SINGLE, single_file.filename)
                single_file.save(save_path)
                uploaded_files.append(url_for('uploaded_file_single', filename=single_file.filename))

    if 'multiple_files' in request.files:
        clear_folder(UPLOAD_FOLDER_MULTIPLE)
        multiple_files = request.files.getlist('multiple_files')
        for file in multiple_files:
            if file:

                # 获取文件名和扩展名
                filename, file_ext = os.path.splitext(file.filename)
                file_ext = file_ext.lower()
                if file_ext == '.tif':
                    # 如果是 .tif 文件，转换为 .png
                    img = Image.open(file)
                    new_filename = filename + '.png'
                    save_path = os.path.join(UPLOAD_FOLDER_MULTIPLE, new_filename)
                    img.save(save_path, format='PNG')
                    uploaded_files.append(url_for('uploaded_file_multiple', filename=new_filename))
                else:
                    # 否则按原格式保存
                    save_path = os.path.join(UPLOAD_FOLDER_MULTIPLE, file.filename)
                    file.save(save_path)
                    uploaded_files.append(url_for('uploaded_file_multiple', filename=file.filename))

    return jsonify({"uploaded_files": uploaded_files})


def detect_main_color_channel(image):
    image_np = np.array(image)
    if image_np.ndim != 3 or image_np.shape[2] != 3:
        raise ValueError("Image is not in expected RGB format.")
    sums = np.sum(np.sum(image_np, axis=0), axis=0)
    main_channel = np.argmax(sums)
    return main_channel


from PIL import Image
import numpy as np


def analyze_fluorescence(image, box):
    # 裁剪出感兴趣区域（ROI）
    roi = image.crop(box)

    # 将ROI转换为NumPy数组
    roi_np = np.array(roi)

    # 计算总荧光强度
    total_fluorescence = np.sum(roi_np)

    # 计算荧光区域内的非零像素数
    fluorescence_area = np.count_nonzero(roi_np)

    # 计算平均荧光强度
    avg_fluorescence = total_fluorescence / fluorescence_area if fluorescence_area > 0 else 0

    # 返回总荧光强度和格式化为两位小数的平均荧光强度
    return int(total_fluorescence), round(avg_fluorescence, 2)


def process_image(file_path, info):
    global color
    try:
        image = Image.open(file_path)
        font = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(2e-2 * image.size[1] + 0.5).astype('int32'))
        colorNum = detect_main_color_channel(Image.open(file_path))
        color = ["red", "green", "blue"][colorNum]

        image2 = Image.open(file_path).convert('L')
    except IOError as e:
        print(f"Could not open image {file_path}: {e}")
        return None, None

    draw = ImageDraw.Draw(image)
    results = []
    for label, identifier, box in info:
        if label == "S":
            left, top, right, bottom = box
            draw.rectangle([left, top, right, bottom], outline="white", width=1)
            text_position = (left, top - 10)
            draw.text(text_position, f'{label}{identifier}', fill="white", font=font)
            total_fluorescence, avg_fluorescence = analyze_fluorescence(image2, box)
            results.append((color, identifier, total_fluorescence, avg_fluorescence))
    del draw
    return image, results

def process_folder(folder_path, info):
    color_results = {'red': [], 'green': [], 'blue': []}
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
            file_path = os.path.join(folder_path, filename)
            processed_image, fluorescence_data = process_image(file_path, info)
            if processed_image:
                processed_image.save(os.path.join(PROCESSED_FOLDER, f"processed_{filename}"))
                sorted_fluorescence_data = sorted(fluorescence_data, key=lambda x: x[2], reverse=True)
                for color, identifier, total_fluorescence, avg_fluorescence in sorted_fluorescence_data:
                    color_results[color].append((identifier, total_fluorescence, avg_fluorescence))
    return color_results


@app.route('/process', methods=['POST'])
def mainProcess():
    clear_folder(PROCESSED_FOLDER)
    yolo = YOLO()
    results = {'red': [], 'green': [], 'blue': []}

    # Process single file uploads
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
    # Process multiple file uploads
    uploaded_files_multiple = os.listdir(UPLOAD_FOLDER_MULTIPLE)
    color_results = process_folder(UPLOAD_FOLDER_MULTIPLE, info)
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

if __name__ == '__main__':
    app.run(debug=True)
