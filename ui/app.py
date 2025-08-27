import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, render_template, request, redirect, url_for
from src.face_module import register_face, verify_face

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', message='Welcome to AI Proctoring System')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        student_id = request.form['student_id']
        success = register_face(student_id)
        return render_template('result.html', action='Registration', success=success, student_id=student_id)
    return render_template('register.html')

@app.route('/verify', methods=['GET', 'POST'])
def verify():
    if request.method == 'POST':
        student_id = request.form['student_id']
        success = verify_face(student_id)
        return render_template('result.html', action='Verification', success=success, student_id=student_id)
    return render_template('verify.html')

if __name__ == '__main__':
    app.run(debug=True)
