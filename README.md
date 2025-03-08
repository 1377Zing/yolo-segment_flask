**yolo-segment_flask**

构建了一个基于Flask 的Web应用，用于处理图像上传、目标检测以及荧光分析任务
框架与库：Flask 搭建 Web 服务，Flask - CORS 解决跨域问题，PIL、OpenCV 和 scikit - image 用于图像处理，NumPy 处理数值计算，YOLO模型来自自定义的yolo模块。

**安装与部署**

**环境要求**

python3.7， torch==1.7.1+cpu
**安装步骤**

**克隆项目仓库：**

git clone https://github.com/1377Zing/yolo-segment_flask.git

cd yolo-segment_flask

pip install -r requirements.txt

python app.py
