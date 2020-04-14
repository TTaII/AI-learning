import os
from flask import Flask, render_template, request
import cv2
import json
import base64
import numpy as np
import tensorflow as tf
import face_recognition
import MTCNN

count = 0
i = 0
frequency = ['0', '0', '0', '0', '0', '0', '0']
known_face_encodings = [0]
person = [frequency.copy()]
temp_fre = [0 for _ in range(7)]
person_num = 0

app = Flask(__name__)
basedir = os.path.abspath(os.path.dirname(__file__))

config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
sess = tf.Session(config=config)
with tf.gfile.FastGFile('model.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')
sess.run(tf.global_variables_initializer())
image = sess.graph.get_tensor_by_name('input_1:0')
output = sess.graph.get_tensor_by_name('predictions/Softmax:0')
Emotion = {0: '生气', 1: '厌恶', 2: '恐惧', 3: '开心', 4: '伤心', 5: '惊讶', 6: '自然'}

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template('echart.html')


@app.route("/loading_face", methods=["POST"])
def loading_face():
    global i, temp_fre, count, person_num
    face_bgr = json.loads(request.data)['image']
    face_bgr = base64.b64decode(str(face_bgr))
    face_bgr = np.fromstring(face_bgr, np.uint8)
    face_bgr = cv2.imdecode(face_bgr, cv2.IMREAD_COLOR)
    cv2.imwrite('Unknown.jpg', face_bgr)
    if person_num == 0:
        return
    unknown_image = face_recognition.load_image_file("Unknown.jpg")
    unknown_image = cv2.resize(unknown_image, (300, 300), interpolation=cv2.INTER_AREA)
    cv2.imwrite('Unknown.jpg', unknown_image)
    try:
        unknown_face_encoding = face_recognition.face_encodings(unknown_image)[0]
        results = face_recognition.compare_faces(known_face_encodings, unknown_face_encoding, tolerance=0.44)
        print(results)
        if not True in results:
            return
        else:
            count += 1
    except IndexError:
        print('No face')
    face_gary = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    face_gary = cv2.resize(face_gary, (64, 64), interpolation=cv2.INTER_AREA)
    face_gary = face_gary[np.newaxis, :, :, np.newaxis]
    Prediction = sess.run(output, feed_dict={image: face_gary})
    Prediction[0][2] = Prediction[0][2] * 0.6
    Prediction[0][0] = Prediction[0][0] * 0.8
    Prediction[0][4] = Prediction[0][4] * 0.8
    P = np.argmax(Prediction[0])
    person[0][P] = str(int(person[0][P]) + 1)
    # temp_fre[P] += 1
    # i += 1
    # if i == 8 and count >= 6:
    #     i = 0
    #     count = 0
    #     for i in range(len(temp_fre)):
    #         person[0][i] = str(int(person[0][i]) + temp_fre[i])
    #     temp_fre = [0 for _ in range(7)]
    return


@app.route('/up_photo', methods=['post'])
def up_photo():
    img = request.files.get('photo')
    if not allowed_file(img.filename):
        return '''
        <!doctype html>
        <title>Not a valid format picture!</title>
        <h1>You should upload a valid picture with face!</h1>
        '''
    path = basedir + "/static/photo/"
    file_path = path + img.filename
    img.save(file_path)

    temp_image = face_recognition.load_image_file(file_path)
    rgb_image = cv2.imread(file_path)
    face_locations = face_recognition.face_locations(rgb_image)
    try:
        (top, right, bottom, left) = face_locations[0]
        temp_image = temp_image[top:bottom, left:right]
        rgb_image = rgb_image[top:bottom, left:right]
        print(face_locations[0])
    except IndexError:
        print(face_locations)
    temp_image = cv2.resize(temp_image, (300, 300), interpolation=cv2.INTER_AREA)
    try:
        face_encodings = face_recognition.face_encodings(temp_image)[0]
    except IndexError:
        return '''
        <!doctype html>
        <title>There is no face in the picture!</title>
        <h1>You should upload a valid picture with face!</h1>
        '''
    known_face_encodings[0] = face_encodings
    person[0] = frequency.copy()
    global person_num
    person_num = 1

    face_path = os.path.join('static/images', request.args.get('filename'))
    cv2.imwrite(face_path, rgb_image)
    return render_template('echart.html', user_image=face_path)


@app.route('/getdata')
def get_data():
    id = int(request.args.get('id'))
    emotion = ['anger', 'disgust', 'fear', 'happy', 'sad', 'surprised', 'normal']
    try:
        fre = person[id]
    except IndexError:
        return json.dumps({'emotion': emotion, 'time': frequency}, ensure_ascii=False)
    return json.dumps({'emotion': emotion, 'time': fre}, ensure_ascii=False)


if __name__ == "__main__":
    app.run("0.0.0.0", 5000, debug=False)
