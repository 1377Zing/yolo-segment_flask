from PIL import ImageFont
from PIL import ImageDraw
import numpy as np
from PIL import Image
from skimage import measure
import cv2


def detect_main_color_channel(image):
    image_np = np.array(image)
    if image_np.ndim != 3 or image_np.shape[2] != 3:
        raise ValueError("Image is not in expected RGB format.")
    sums = np.sum(np.sum(image_np, axis=0), axis=0)
    main_channel = np.argmax(sums)
    return main_channel

def detect_color(image):
    color_ranges = {
        0: {'lower': np.array([0, 0, 100]), 'upper': np.array([50, 50, 255])},
        1: {'lower': np.array([0, 100, 0]), 'upper': np.array([50, 255, 50])},
        2: {'lower': np.array([100, 0, 0]), 'upper': np.array([255, 50, 50])},
        3: {'lower': np.array([0, 100, 100]), 'upper': np.array([50, 255, 255])}
    }

    image_np = np.array(image)

    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)


    for color_id, color_range in color_ranges.items():
        mask = cv2.inRange(image_bgr, color_range['lower'], color_range['upper'])
        if cv2.countNonZero(mask) > 0:
            return color_id

    return None

def analyze_fluorescence(image, bbox):
    left, top, right, bottom = bbox
    image_np = np.array(image)

    for row in range(top, bottom):
        for col in range(left, right):
            pixel_value = image_np[row, col]

    roi_np = image_np[top:bottom, left:right]
    total_fluorescence = np.sum(roi_np)

    fluorescence_area = np.count_nonzero(roi_np)

    avg_fluorescence = total_fluorescence / fluorescence_area if fluorescence_area > 0 else 0

    return int(fluorescence_area), round(avg_fluorescence, 2)


def analyze_fluorescence2(image, box1,box2):
    roi1 = image.crop(box1)
    roi2 = image.crop(box2)
    roi_np1 = np.array(roi1)
    roi_np2 = np.array(roi2)
    total_fluorescence1 = np.sum(roi_np1)
    total_fluorescence2 = np.sum(roi_np2)
    total_fluorescence = total_fluorescence1-total_fluorescence2
    fluorescence_area1 = np.count_nonzero(roi_np1)
    fluorescence_area2 = np.count_nonzero(roi_np2)
    fluorescence_area = fluorescence_area1-fluorescence_area2

    avg_fluorescence = total_fluorescence / fluorescence_area if fluorescence_area > 0 else 0

    return int(total_fluorescence), round(avg_fluorescence, 2)

def is_intersecting(box1, box2):
    return not (box1[2] < box2[0] or box1[0] > box2[2] or box1[3] < box2[1] or box1[1] > box2[3])
def calculate_intersection_area(box1, box2):
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0
    return (x_right - x_left) * (y_bottom - y_top)

def analyze_fluorescence21(image, yolobox, coords):
    left, top, right, bottom = yolobox
    image_np = np.array(image)
    rows, cols = coords[:, 0], coords[:, 1]

    mask = ~((left < cols) & (cols < right) & (top < rows) & (rows < bottom))
    remaining_rows = rows[mask]
    remaining_cols = cols[mask]

    remaining_pixels = image_np[remaining_rows, remaining_cols]

    remaining_nonzero_count = np.count_nonzero(remaining_pixels)

    remaining_pixel_sum = np.sum(remaining_pixels)

    avg_fluorescence = remaining_pixel_sum / remaining_nonzero_count if remaining_nonzero_count > 0 else 0

    return int(remaining_nonzero_count), round(avg_fluorescence, 2)

def is_intersecting1(yolobox, convex_image):
    left, top, right, bottom = yolobox
    for coord in convex_image:
        row, col = coord
        if left <= col <= right and top <= row <= bottom:
            return True
    return False
def calculate_intersection_area1(yolobox, convex_image):
    left, top, right, bottom = yolobox
    intersection_count = 0

    for coord in convex_image:
        row, col = coord
        if left <= col <= right and top <= row <= bottom:
            intersection_count += 1
    return intersection_count

def print_seg(image_path, info, mode):
    original_image = Image.open(image_path)
    if original_image.mode != 'RGB':
        original_image = original_image.convert('RGB')

    original_np = np.array(original_image, dtype=np.uint8)
    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    colorNum = detect_color(image)
    color = ["red", "green", "blue", "yellow"][colorNum]
    tolerance = 200
    lower_white = np.array([255 - tolerance, 255 - tolerance, 255 - tolerance], dtype=np.uint8)
    upper_white = np.array([255, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(original_np, lower_white, upper_white)
    original_np[mask > 0] = [0, 0, 0]

    original_image = Image.fromarray(original_np)

    image2 = original_image.convert('L')

    fluorescence_channel_np = np.array(image2)
    alpha = 5
    beta = -2
    adjusted_image = cv2.convertScaleAbs(fluorescence_channel_np, alpha=alpha, beta=beta)
    manual_threshold = 3
    _, binary_image = cv2.threshold(fluorescence_channel_np, manual_threshold, 255, cv2.THRESH_BINARY)
    labeled_image = measure.label(binary_image)
    regions = measure.regionprops(labeled_image, intensity_image=fluorescence_channel_np)
    selected_regions = [region for region in regions if 20 <= region.area <= 1000]
    result = []

    font_path = 'model_data/simhei.ttf'
    font = ImageFont.truetype(font=font_path, size=np.floor(2e-2 * original_image.size[1] + 0.5).astype('int32'))
    draw = ImageDraw.Draw(original_image)
    for max_region in selected_regions :
                convex_image = max_region.convex_image.astype(np.uint8) * 255

                contours, _ = cv2.findContours(convex_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                bbox_min_row, bbox_min_col, bbox_max_row, bbox_max_col = max_region.bbox
                for contour in contours:
                    contour_points = []
                    for point in contour:
                        x = point[0][0] + bbox_min_col
                        y = point[0][1] + bbox_min_row
                        contour_points.extend([x, y])
                    if len(contour_points) > 2:
                            draw.polygon(contour_points, outline='red', width=1)


                result_sorted = sorted(result, key=lambda x: x["id"])




    return original_image, result_sorted
def process_tail_image(image_path, info, mode):
    original_image = Image.open(image_path)
    if original_image.mode != 'RGB':
        original_image = original_image.convert('RGB')

    original_np = np.array(original_image, dtype=np.uint8)
    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    colorNum = detect_color(image)
    color = ["red", "green", "blue", "yellow"][colorNum]
    tolerance = 200
    lower_white = np.array([255 - tolerance, 255 - tolerance, 255 - tolerance], dtype=np.uint8)
    upper_white = np.array([255, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(original_np, lower_white, upper_white)
    original_np[mask > 0] = [0, 0, 0]

    original_image = Image.fromarray(original_np)

    image2 = original_image.convert('L')

    fluorescence_channel_np = np.array(image2)
    if mode == 0:
        equalized_image = cv2.equalizeHist(fluorescence_channel_np)
        manual_threshold = 30
        _, binary_image = cv2.threshold(equalized_image, manual_threshold, 255, cv2.THRESH_BINARY)

        labeled_image = measure.label(binary_image)
        regions = measure.regionprops(labeled_image, intensity_image=fluorescence_channel_np)
        selected_regions = [region for region in regions if 20 <= region.area <= 1000]
        result = []

        font_path = 'model_data/simhei.ttf'
        font = ImageFont.truetype(font=font_path, size=np.floor(2e-2 * original_image.size[1] + 0.5).astype('int32'))
        draw = ImageDraw.Draw(original_image)

        for label, identifier, info_box in info:
            intersecting_regions = [
                region for region in selected_regions if
                is_intersecting1(info_box, region.coords)
            ]

            if len(intersecting_regions) == 1:
                region = intersecting_regions[0]
                bbox_pil = (region.bbox[1], region.bbox[0], region.bbox[3], region.bbox[2])


                info_fluorescence = analyze_fluorescence(image2, info_box)
                remaining_fluorescence = analyze_fluorescence21(image2, info_box, region.coords)


                left, top, right, bottom = info_box
                draw.rectangle([left, top, right, bottom], outline="white", width=1)
                text_position = (left, top - 10)
                draw.text(text_position, f'{label}{identifier}', fill="white", font=font)

                convex_image = region.convex_image.astype(np.uint8) * 255

                contours, _ = cv2.findContours(convex_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                bbox_min_row, bbox_min_col, bbox_max_row, bbox_max_col = region.bbox
                for contour in contours:
                    contour_points = []
                    for point in contour:
                        x = point[0][0] + bbox_min_col
                        y = point[0][1] + bbox_min_row
                        contour_points.extend([x, y])
                    if len(contour_points) > 2:
                        draw.polygon(contour_points, outline='red', width=1)

                result.append({
                    "id": identifier,
                    "color": color,
                    "head": {
                        "total_fluorescence": info_fluorescence[0],
                        "avg_fluorescence": info_fluorescence[1]
                    },
                    "tail": {
                        "total_fluorescence": remaining_fluorescence[0],
                        "avg_fluorescence": remaining_fluorescence[1]
                    }
                })

            elif len(intersecting_regions) > 1:
                max_region = max(intersecting_regions,
                                 key=lambda region: calculate_intersection_area1(info_box, region.coords))


                info_fluorescence = analyze_fluorescence(image2, info_box)
                remaining_fluorescence = analyze_fluorescence21(image2, info_box, max_region.coords)

                left, top, right, bottom = info_box
                draw.rectangle([left, top, right, bottom], outline="white", width=1)
                text_position = (left, top - 10)
                draw.text(text_position, f'{label}{identifier}', fill="white", font=font)

                convex_image = max_region.convex_image.astype(np.uint8) * 255

                contours, _ = cv2.findContours(convex_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                bbox_min_row, bbox_min_col, bbox_max_row, bbox_max_col = max_region.bbox
                for contour in contours:
                    contour_points = []
                    for point in contour:
                        x = point[0][0] + bbox_min_col
                        y = point[0][1] + bbox_min_row
                        contour_points.extend([x, y])
                    if len(contour_points) > 2:
                        draw.polygon(contour_points, outline='red', width=1)

                result.append({
                    "id": identifier,
                    "color": color,
                    "head": {
                        "total_fluorescence": info_fluorescence[0],
                        "avg_fluorescence": info_fluorescence[1]
                    },
                    "tail": {
                        "total_fluorescence": remaining_fluorescence[0],
                        "avg_fluorescence": remaining_fluorescence[1]
                    }
                })

        result_sorted = sorted(result, key=lambda x: x["id"])
    elif mode == 1:
        # alpha = 5
        # beta = -2
        # adjusted_image = cv2.convertScaleAbs(fluorescence_channel_np, alpha=alpha, beta=beta)
        manual_threshold = 3
        _, binary_image = cv2.threshold(fluorescence_channel_np, manual_threshold, 255, cv2.THRESH_BINARY)
        labeled_image = measure.label(binary_image)
        regions = measure.regionprops(labeled_image, intensity_image=fluorescence_channel_np)
        selected_regions = [region for region in regions if 20 <= region.area <= 1000]
        result = []

        font_path = 'model_data/simhei.ttf'
        font = ImageFont.truetype(font=font_path, size=np.floor(2e-2 * original_image.size[1] + 0.5).astype('int32'))
        draw = ImageDraw.Draw(original_image)

        for label, identifier, info_box in info:
            intersecting_regions = [
                region for region in selected_regions if
                is_intersecting1(info_box, region.coords)
            ]

            if len(intersecting_regions) == 1:
                region = intersecting_regions[0]

                info_fluorescence = analyze_fluorescence(image2, info_box)
                remaining_fluorescence = analyze_fluorescence21(image2, info_box, region.coords)


                left, top, right, bottom = info_box
                draw.rectangle([left, top, right, bottom], outline="white", width=1)
                text_position = (left, top - 10)
                draw.text(text_position, f'{label}{identifier}', fill="white", font=font)

                convex_image = region.convex_image.astype(np.uint8) * 255

                contours, _ = cv2.findContours(convex_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                bbox_min_row, bbox_min_col, bbox_max_row, bbox_max_col = region.bbox
                for contour in contours:
                    contour_points = []
                    for point in contour:
                        x = point[0][0] + bbox_min_col
                        y = point[0][1] + bbox_min_row
                        contour_points.extend([x, y])
                    if len(contour_points) > 2:
                        draw.polygon(contour_points, outline='red', width=1)

                result.append({
                    "id": identifier,
                    "color": color,
                    "head": {
                        "total_fluorescence": info_fluorescence[0],
                        "avg_fluorescence": info_fluorescence[1]
                    },
                    "tail": {
                        "total_fluorescence": remaining_fluorescence[0],
                        "avg_fluorescence": remaining_fluorescence[1]
                    }
                })

            elif len(intersecting_regions) > 1:

                max_region = max(intersecting_regions,
                                 key=lambda region: calculate_intersection_area1(info_box, region.coords))

                info_fluorescence = analyze_fluorescence(image2, info_box)
                remaining_fluorescence = analyze_fluorescence21(image2, info_box, max_region.coords)

                left, top, right, bottom = info_box
                draw.rectangle([left, top, right, bottom], outline="white", width=1)
                text_position = (left, top - 10)
                draw.text(text_position, f'{label}{identifier}', fill="white", font=font)

                convex_image = max_region.convex_image.astype(np.uint8) * 255

                contours, _ = cv2.findContours(convex_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                bbox_min_row, bbox_min_col, bbox_max_row, bbox_max_col = max_region.bbox
                for contour in contours:
                    contour_points = []
                    for point in contour:
                        x = point[0][0] + bbox_min_col
                        y = point[0][1] + bbox_min_row
                        contour_points.extend([x, y])
                    if len(contour_points) > 2:
                        draw.polygon(contour_points, outline='red', width=1)

                result.append({
                    "id": identifier,
                    "color": color,
                    "head": {
                        "total_fluorescence": info_fluorescence[0],
                        "avg_fluorescence": info_fluorescence[1]
                    },
                    "tail": {
                        "total_fluorescence": remaining_fluorescence[0],
                        "avg_fluorescence": remaining_fluorescence[1]
                    }
                })


        result_sorted = sorted(result, key=lambda x: x["id"])




    return original_image, result_sorted

def process_image(file_path, info):
    global color
    try:
        image = Image.open(file_path)
        font = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(2e-2 * image.size[1] + 0.5).astype('int32'))
        image2 = Image.open(file_path)
        if image2.mode != 'RGB':
            image2 = image2.convert('RGB')
        colorNum = detect_color(image2)
        color = ["red", "green", "blue", "yellow"][colorNum]

        image2 = Image.open(file_path).convert('L')
    except IOError as e:
        print(f"Could not open image {file_path}: {e}")
        return None, None

    draw = ImageDraw.Draw(image)
    results = []
    for label, identifier, box in info:
        if label == "S":
            left, top, right, bottom = box
            draw.rectangle([left, top, right, bottom], outline=(255, 255, 255), width=1)
            text_position = (left, top - 10)
            draw.text(text_position, f'{label}{identifier}', fill="white", font=font)
            total_fluorescence, avg_fluorescence = analyze_fluorescence(image2, box)
            results.append((color, identifier, total_fluorescence, avg_fluorescence))
    del draw
    return image, results

