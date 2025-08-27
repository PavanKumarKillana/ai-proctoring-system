from flask import Flask, render_template, request, redirect, url_for
from src.face_module import register_face, verify_face

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        student_id = request.form['student_id']
        if not student_id:
            return redirect(url_for('result', message='Error: Student ID cannot be empty.'))
        success = register_face(student_id)
        if success:
            return redirect(url_for('result', message=f'Registration successful for {student_id}!'))
        else:
            return redirect(url_for('result', message=f'Registration failed for {student_id}. Ensure one face is visible, use bright lighting, and center your face.'))
    return render_template('register.html')

@app.route('/verify', methods=['GET', 'POST'])
def verify():
    if request.method == 'POST':
        student_id = request.form['student_id']
        if not student_id:
            return redirect(url_for('result', message='Error: Student ID cannot be empty.'))
        success = verify_face(student_id)
        if success:
            return redirect(url_for('result', message=f'Verification successful for {student_id}!'))
        else:
            return redirect(url_for('result', message=f'Verification failed for {student_id}. Ensure one face is visible or register first.'))
    return render_template('verify.html')

@app.route('/result')
def result():
    message = request.args.get('message', 'No message provided.')
    return render_template('result.html', message=message)

if __name__ == '__main__':
    app.run(debug=True)