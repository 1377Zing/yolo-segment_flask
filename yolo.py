import colorsys
import os
import time
import numpy as np
import torch
import torch.nn as nn
from PIL import ImageDraw, ImageFont
from nets.yolo import YoloBody
from utils.utils import (cvtColor, get_classes, preprocess_input,
                      resize_image, show_config)
from utils.utils_bbox import DecodeBox

# Notes for training your own dataset!
class YOLO(object):
    _defaults = {
        # Modify model_path and classes_path when using your own trained model.
        # model_path points to the weight file in the logs folder, classes_path points to the txt in model_data.
        # Choose the weight file with lower validation set loss. Lower validation loss doesn't mean higher mAP.
        # If there's a shape mismatch, also check the parameters during training.
        "model_path": './model_data/EVISANS.pth',
        "classes_path": './model_data/cls_classes.txt',
        # Input image size, must be a multiple of 32.
        "input_shape": [640, 640],
        # YOLOv8 version used:
        "phi": 's',
        # Only prediction boxes with scores greater than the confidence will be kept.
        "confidence": 0.3,
        # NMS IoU value for non - maximum suppression.
        "nms_iou": 0.3,
        # Whether to use letterbox_image for distortion - free resizing.
        # After tests, direct resizing works better.
        "letterbox_image": True,
        # Whether to use Cuda. Set to False if no GPU is available.
        "cuda": False,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # Initialize YOLO
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            self._defaults[name] = value

        # Get the number of classes and anchors
        self.class_names, self.num_classes = get_classes(self.classes_path)
        self.bbox_util = DecodeBox(self.num_classes, (self.input_shape[0], self.input_shape[1]))

        # Set different colors for drawing boxes
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        self.generate()

        show_config(**self._defaults)

    # Generate the model
    def generate(self, onnx=False):
        # Build the YOLO model and load its weights
        self.net = YoloBody(self.input_shape, self.num_classes, self.phi)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net = self.net.fuse().eval()
        print('{} model, and classes loaded.'.format(self.model_path))
        if not onnx:
            if self.cuda:
                self.net = nn.DataParallel(self.net)
                self.net = self.net.cuda()

    def detect_image(self, image, crop=False, count=False):
        image_shape = np.array(np.shape(image)[0:2])
        image = cvtColor(image)  # Ensure it's RGB
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)
            results = self.bbox_util.non_max_suppression(outputs, self.num_classes, self.input_shape,
                                                         image_shape, self.letterbox_image, conf_thres=self.confidence,
                                                         nms_thres=self.nms_iou)

            if results[0] is None:
                return image, []

            top_label = np.array(results[0][:, 5], dtype='int32')
            top_boxes = results[0][:, :4]

        boxes_info = []
        identifier = 1  # Unique identifier for each 'S' object

        font = ImageFont.truetype(font='model_data/simhei.ttf',
                                  size=np.floor(2e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = max(int((image.size[0] + image.size[1]) // 1000), 1)  # Reduce thickness proportionally
        draw = ImageDraw.Draw(image)

        for i, c in enumerate(top_label):
            predicted_class = self.class_names[c]
            if predicted_class == 'S':  # Filter to include only 'S' objects
                box = top_boxes[i]
                top, left, bottom, right = box
                top = max(0, np.floor(top).astype('int32'))
                left = max(0, np.floor(left).astype('int32'))
                bottom = min(image.size[1], np.floor(bottom).astype('int32'))
                right = min(image.size[0], np.floor(right).astype('int32'))

                label = f'{predicted_class}{identifier}'
                label_size = draw.textsize(label, font)
                text_origin = np.array([left, top - label_size[1]])

                draw.rectangle([left, top, right, bottom], outline=(255, 255, 255), width=1)
                text_position = (left, top - 10)
                draw.text(text_position, f'{label}', fill="white", font=font)

                boxes_info.append((predicted_class, identifier, (left, top, right, bottom)))
                identifier += 1  # Increment identifier for the next 'S' object

        del draw
        return image, boxes_info

    def get_FPS(self, image, test_interval):
        image_shape = np.array(np.shape(image)[0:2])
        # Convert the image to RGB to prevent errors for grayscale images.
        # Only RGB images are supported.
        image = cvtColor(image)
        # Add gray bars to the image for distortion - free resizing.
        # Direct resizing can also be used.
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        # Add the batch_size dimension
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            # Input the image into the network for prediction!
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)
            # Stack the prediction boxes and perform non - maximum suppression
            results = self.bbox_util.non_max_suppression(outputs, self.num_classes, self.input_shape,
                                                         image_shape, self.letterbox_image, conf_thres=self.confidence,
                                                         nms_thres=self.nms_iou)

        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                # Input the image into the network for prediction!
                outputs = self.net(images)
                outputs = self.bbox_util.decode_box(outputs)
                # Stack the prediction boxes and perform non - maximum suppression
                results = self.bbox_util.non_max_suppression(outputs, self.num_classes, self.input_shape,
                                                             image_shape, self.letterbox_image,
                                                             conf_thres=self.confidence, nms_thres=self.nms_iou)

        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    def detect_heatmap(self, image, heatmap_save_path):
        import cv2
        import matplotlib.pyplot as plt

        def sigmoid(x):
            y = 1.0 / (1.0 + np.exp(-x))
            return y

        # Convert the image to RGB to prevent errors for grayscale images.
        # Only RGB images are supported.
        image = cvtColor(image)
        # Add gray bars to the image for distortion - free resizing.
        # Direct resizing can also be used.
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        # Add the batch_size dimension
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            # Input the image into the network for prediction!
            dbox, cls, x, anchors, strides = self.net(images)
            outputs = [xi.split((xi.size()[1] - self.num_classes, self.num_classes), 1)[1] for xi in x]

        plt.imshow(image, alpha=1)
        plt.axis('off')
        mask = np.zeros((image.size[1], image.size[0]))
        for sub_output in outputs:
            sub_output = sub_output.cpu().numpy()
            b, c, h, w = np.shape(sub_output)
            sub_output = np.transpose(np.reshape(sub_output, [b, -1, h, w]), [0, 2, 3, 1])[0]
            score = np.max(sigmoid(sub_output[..., :]), -1)
            score = cv2.resize(score, (image.size[0], image.size[1]))
            normed_score = (score * 255).astype('uint8')
            mask = np.maximum(mask, normed_score)

        plt.imshow(mask, alpha=0.5, interpolation='nearest', cmap="jet")

        plt.axis('off')
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(heatmap_save_path, dpi=200, bbox_inches='tight', pad_inches=-0.1)
        print("Save to the " + heatmap_save_path)
        plt.show()

    def convert_to_onnx(self, simplify, model_path):
        import onnx
        self.generate(onnx=True)

        im = torch.zeros(1, 3, *self.input_shape).to('cpu')  # image size(1, 3, 512, 512) BCHW
        input_layer_names = ["images"]
        output_layer_names = ["output"]

        # Export the model
        print(f'Starting export with onnx {onnx.__version__}.')
        torch.onnx.export(self.net,
                          im,
                          f=model_path,
                          verbose=False,
                          opset_version=12,
                          training=torch.onnx.TrainingMode.EVAL,
                          do_constant_folding=True,
                          input_names=input_layer_names,
                          output_names=output_layer_names,
                          dynamic_axes=None)

        # Checks
        model_onnx = onnx.load(model_path)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model

        # Simplify onnx
        if simplify:
            import onnxsim
            print(f'Simplifying with onnx - simplifier {onnxsim.__version__}.')
            model_onnx, check = onnxsim.simplify(
                model_onnx,
                dynamic_input_shape=False,
                input_shapes=None)
            assert check, 'assert check failed'
            onnx.save(model_onnx, model_path)

        print('Onnx model save as {}'.format(model_path))

    def get_map_txt(self, image_id, image, class_names, map_out_path):
        f = open(os.path.join(map_out_path, "detection - results/" + image_id + ".txt"), "w", encoding='utf-8')
        image_shape = np.array(np.shape(image)[0:2])
        # Convert the image to RGB to prevent errors for grayscale images.
        # Only RGB images are supported.
        image = cvtColor(image)
        # Add gray bars to the image for distortion - free resizing.
        # Direct resizing can also be used.
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        # Add the batch_size dimension
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            # Input the image into the network for prediction!
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)
            # Stack the prediction boxes and perform non - maximum suppression
            results = self.bbox_util.non_max_suppression(outputs, self.num_classes, self.input_shape,
                                                         image_shape, self.letterbox_image, conf_thres=self.confidence,
                                                         nms_thres=self.nms_iou)

            if results[0] is None:
                return

            top_label = np.array(results[0][:, 5], dtype='int32')
            top_conf = results[0][:, 4]
            top_boxes = results[0][:, :4]

        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = str(top_conf[i])

            top, left, bottom, right = box
            if predicted_class not in class_names:
                continue

            f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)),
                                             str(int(bottom))))

        f.close()
        return