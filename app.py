from flask import Flask, render_template, Response, redirect, url_for
from recognizer.recognizer import SimpleSignLanguageRecognizer

app = Flask(__name__)

recognizer = SimpleSignLanguageRecognizer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

def gen_frames():
    while True:
        frame = recognizer.get_frame()
        if frame is None:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
