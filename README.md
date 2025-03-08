### yolo-segment_flask

A Flask-based web application is built to handle tasks such as image upload, object detection, and fluorescence analysis.

**Frameworks and Libraries**: Flask is used to build the web service, Flask-CORS is used to solve cross - domain issues, PIL, OpenCV, and scikit - image are used for image processing, NumPy is used for numerical computation, and the YOLO model comes from a custom `yolo` module.

### Installation and Deployment

#### Environment Requirements
Python 3.7, torch==1.7.1+cpu

#### Installation Steps
```bash
git clone https://github.com/1377Zing/yolo-segment_flask.git
cd yolo-segment_flask
pip install -r requirements.txt
python app.py
```
