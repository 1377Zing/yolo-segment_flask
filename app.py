from flask import Flask, render_template, request, send_from_directory, url_for, jsonify
from PIL import ImageFont
from flask_cors import CORS
from yolo import YOLO
from PIL import ImageDraw
import os
import numpy as np
from PIL import Image
from skimage import measure
import cv2
app = Flask(__name__)
CORS(app)
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

def analyze_fluorescence2(image, box1,box2):
    # 裁剪出感兴趣区域（ROI）
    roi1 = image.crop(box1)
    roi2 = image.crop(box2)
    # 将ROI转换为NumPy数组
    roi_np1 = np.array(roi1)
    roi_np2 = np.array(roi2)
    # 计算总荧光强度
    total_fluorescence1 = np.sum(roi_np1)
    total_fluorescence2 = np.sum(roi_np2)
    total_fluorescence = total_fluorescence1-total_fluorescence2
    # 计算荧光区域内的非零像素数
    fluorescence_area1 = np.count_nonzero(roi_np1)
    fluorescence_area2 = np.count_nonzero(roi_np2)
    fluorescence_area = fluorescence_area1-fluorescence_area2

    # 计算平均荧光强度
    avg_fluorescence = total_fluorescence / fluorescence_area if fluorescence_area > 0 else 0

    # 返回总荧光强度和格式化为两位小数的平均荧光强度
    return int(total_fluorescence), round(avg_fluorescence, 2)

def is_intersecting(box1, box2):
    return not (box1[2] < box2[0] or box1[0] > box2[2] or box1[3] < box2[1] or box1[1] > box2[3])
def calculate_intersection_area(box1, box2):
    """
    计算两个矩形区域的相交面积
    """
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0  # 没有相交
    return (x_right - x_left) * (y_bottom - y_top)


def process_tail_image(image_path, info):
    original_image = Image.open(image_path)
    if original_image.mode != 'RGB':
        original_image = original_image.convert('RGB')
    image2 = Image.open(image_path).convert('L')#灰度图
    original_np = np.array(original_image, dtype=np.uint8)
    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    colorNum = detect_main_color_channel(image)
    color = ["red", "green", "blue"][colorNum]
    tolerance = 200
    lower_white = np.array([255 - tolerance, 255 - tolerance, 255 - tolerance], dtype=np.uint8)
    upper_white = np.array([255, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(original_np, lower_white, upper_white)
    original_np[mask > 0] = [0, 0, 0]

    # 更新 original_image 以反映修改
    original_image = Image.fromarray(original_np)

    fluorescence_channel = original_image.split()[colorNum]
    fluorescence_channel_np = np.array(fluorescence_channel)

    equalized_image = cv2.equalizeHist(fluorescence_channel_np)
    manual_threshold = 30
    _, binary_image = cv2.threshold(equalized_image, manual_threshold, 255, cv2.THRESH_BINARY)

    labeled_image = measure.label(binary_image)
    regions = measure.regionprops(labeled_image, intensity_image=fluorescence_channel_np)
    selected_regions = [region for region in regions if region.area > 50]

    result = []

    font_path = 'model_data/simhei.ttf'  # 更新字体路径
    font = ImageFont.truetype(font=font_path, size=np.floor(2e-2 * original_image.size[1] + 0.5).astype('int32'))
    draw = ImageDraw.Draw(original_image)

    # 遍历每个 info，找到与其相交的所有 regions
    for label, identifier, info_box in info:
        # 找出所有与当前 info_box 相交的 regions
        intersecting_regions = [
            region for region in selected_regions if
            is_intersecting(info_box, (region.bbox[1], region.bbox[0], region.bbox[3], region.bbox[2]))
        ]

        # 如果只有一个 region 与 info 相交，且该 region 只与一个 info 相交
        if len(intersecting_regions) == 1 and len([inf for inf in info if
                                                   is_intersecting((intersecting_regions[0].bbox[1],
                                                                    intersecting_regions[0].bbox[0],
                                                                    intersecting_regions[0].bbox[3],
                                                                    intersecting_regions[0].bbox[2]), inf[2])]) == 1:
            region = intersecting_regions[0]
            bbox_pil = (region.bbox[1], region.bbox[0], region.bbox[3], region.bbox[2])

            # 分析荧光数据
            info_fluorescence = analyze_fluorescence(image2, info_box)
            remaining_fluorescence = analyze_fluorescence2(image2, bbox_pil, info_box)

            # 绘制矩形框并添加到结果
            left, top, right, bottom = info_box
            draw.rectangle([left, top, right, bottom], outline="white", width=1)
            text_position = (left, top - 10)
            draw.text(text_position, f'{label}{identifier}', fill="white", font=font)

            draw.rectangle(bbox_pil, outline="red", width=2)
            result.append({
                "id": identifier,
                "color": color,
                "region_bbox": bbox_pil,
                "region_area": region.area,
                "head": {
                    "total_fluorescence": info_fluorescence[0],
                    "avg_fluorescence": info_fluorescence[1]
                },
                "tail": {
                    "total_fluorescence": remaining_fluorescence[0],
                    "avg_fluorescence": remaining_fluorescence[1]
                }
            })

        # 如果有多个 region 与 info 相交
        elif len(intersecting_regions) > 1:
            # 筛选出只与一个 info 相交的 regions
            single_info_regions = [
                region for region in intersecting_regions
                if len([inf for inf in info if
                        is_intersecting((region.bbox[1], region.bbox[0], region.bbox[3], region.bbox[2]), inf[2])]) == 1
            ]

            # 如果存在只与一个 info 相交的 regions
            if single_info_regions:
                # 选择相交面积最大的 region
                max_region = max(single_info_regions, key=lambda region: calculate_intersection_area(info_box, (
                    region.bbox[1], region.bbox[0], region.bbox[3], region.bbox[2])))

                bbox_pil = (max_region.bbox[1], max_region.bbox[0], max_region.bbox[3], max_region.bbox[2])

                # 分析荧光数据
                info_fluorescence = analyze_fluorescence(image2, info_box)
                remaining_fluorescence = analyze_fluorescence2(image2, bbox_pil, info_box)

                # 绘制矩形框并添加到结果
                left, top, right, bottom = info_box
                draw.rectangle([left, top, right, bottom], outline="white", width=1)
                text_position = (left, top - 10)
                draw.text(text_position, f'{label}{identifier}', fill="white", font=font)

                draw.rectangle(bbox_pil, outline="red", width=2)
                result.append({
                    "id": identifier,
                    "color": color,
                    "region_bbox": bbox_pil,
                    "region_area": max_region.area,
                    "head": {
                        "total_fluorescence": info_fluorescence[0],
                        "avg_fluorescence": info_fluorescence[1]
                    },
                    "tail": {
                        "total_fluorescence": remaining_fluorescence[0],
                        "avg_fluorescence": remaining_fluorescence[1]
                    }
                })

    # 根据 id 对结果排序
    result_sorted = sorted(result, key=lambda x: x["id"])

    return original_image, result_sorted


def process_folder2(folder_path, info):
    color_results = {}
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(
                ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
            file_path = os.path.join(folder_path, filename)
            processed_image, fluorescence_data = process_tail_image(file_path, info)

            if processed_image:
                try:
                    processed_image.save(os.path.join(PROCESSED_FOLDER, f"processed_{filename}"))
                except Exception as e:
                    print(f"Error saving image: {e}")

                for res in fluorescence_data:
                    # 继续处理 item
                    color = res["color"]
                    head_total_fluorescence = res["head"]["total_fluorescence"]
                    head_avg_fluorescence = res["head"]["avg_fluorescence"]
                    tail_total_fluorescence = res["tail"]["total_fluorescence"]
                    tail_avg_fluorescence = res["tail"]["avg_fluorescence"]

                    # 创建颜色分类并添加到 color_results 中
                    if color not in color_results:
                        color_results[color] = []

                    color_results[color].append({
                        "id": res['id'],
                        "head": [head_total_fluorescence, head_avg_fluorescence],
                        "tail": [tail_total_fluorescence, tail_avg_fluorescence]
                    })

    return color_results


def process_image(file_path, info):
    global color
    try:
        image = Image.open(file_path)
        font = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(2e-2 * image.size[1] + 0.5).astype('int32'))
        image2 = Image.open(file_path)
        if image2.mode != 'RGB':
            image2 = image2.convert('RGB')
        colorNum = detect_main_color_channel(image2)
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
                # sorted_fluorescence_data = sorted(fluorescence_data, key=lambda x: x[1], reverse=True)
                for color, identifier, total_fluorescence, avg_fluorescence in fluorescence_data:
                    color_results[color].append((identifier, total_fluorescence, avg_fluorescence))
    return color_results


@app.route('/process', methods=['POST'])
def mainProcess():
    clear_folder(PROCESSED_FOLDER)
    yolo = YOLO()
    results = {'red': [], 'green': [], 'blue': []}
    info = None  # 初始化变量
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


@app.route('/process2', methods=['POST'])
def mainProcess2():
    clear_folder(PROCESSED_FOLDER)
    yolo = YOLO()
    results = {'red': [], 'green': [], 'blue': []}
    info = None  # 初始化变量
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
    color_results = process_folder2(UPLOAD_FOLDER_MULTIPLE, info)
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
