import time
import sqlite3
import sys
import os
import secrets
from flask import (Flask, Response, render_template, request,
                   redirect, url_for, jsonify, session as flask_session)
import cv2
from threading import Thread, Lock

# Ensure the root directory is in sys.path so 'src' can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.face_module import (register_face, verify_face_with_gaze, current_state,
                              init_db, get_activity_log, get_session_summary)
import src.face_module as face_module
from src.webrtc_processor import ExamSession, active_sessions

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

# ===============================
# SECURITY  (Production_2 addition)
# ===============================
# Strong random secret key for session signing.
# In a real deployment this would be an environment variable.
app.secret_key = os.environ.get('SECRET_KEY', secrets.token_hex(32))

# Rate limiting — requires: pip install flask-limiter
try:
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address
    limiter = Limiter(
        key_func=get_remote_address,
        app=app,
        default_limits=["200 per day", "60 per hour"],
        storage_uri="memory://"
    )
    RATE_LIMITING_ENABLED = True
    print("[Security] Rate limiting enabled.")
except ImportError:
    limiter = None
    RATE_LIMITING_ENABLED = False
    print("[Security] flask-limiter not installed — rate limiting disabled. Run: pip install flask-limiter")

# Security headers — requires: pip install flask-talisman
try:
    from flask_talisman import Talisman
    # Allow inline scripts/styles needed by the proctoring UI
    csp = {
        'default-src': ["'self'"],
        'script-src':  ["'self'", "'unsafe-inline'"],
        'style-src':   ["'self'", "'unsafe-inline'", 'fonts.googleapis.com'],
        'font-src':    ["'self'", 'fonts.gstatic.com'],
        'img-src':     ["'self'", 'data:', 'blob:'],
        'media-src':   ["'self'", 'blob:'],
        'connect-src': ["'self'", 'blob:'],
    }
    Talisman(app,
             force_https=False,          # True in production behind HTTPS/ngrok
             strict_transport_security=False,
             content_security_policy=csp,
             frame_options='DENY',       # X-Frame-Options: DENY
             referrer_policy='no-referrer')
    print("[Security] Security headers (flask-talisman) enabled.")
except ImportError:
    print("[Security] flask-talisman not installed — security headers disabled. Run: pip install flask-talisman")


@app.after_request
def add_cache_control(response):
    """Force browsers to never cache pages, so updates show instantly."""
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma']  = 'no-cache'
    response.headers['Expires'] = '0'
    # Extra security headers that work even without flask-talisman
    response.headers.setdefault('X-Content-Type-Options', 'nosniff')
    response.headers.setdefault('X-Frame-Options', 'DENY')
    response.headers.setdefault('X-XSS-Protection', '1; mode=block')
    return response


# ===============================
# EXAM SETTINGS
# ===============================
EXAM_DURATION = 120  # 2 minutes

# ===============================
# DATABASE SETUP
# ===============================
def init_violations_db():
    conn = sqlite3.connect("violations.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS violations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT,
            violation_type TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def save_violation(student_id, violation_type):
    conn = sqlite3.connect("violations.db")
    c = conn.cursor()
    c.execute("INSERT INTO violations (student_id, violation_type) VALUES (?, ?)",
              (student_id, violation_type))
    conn.commit()
    conn.close()

# ===============================
# ACCESS LOG  (NEW — who visits, when, from where)
# ===============================
def init_access_log_db():
    conn = sqlite3.connect("violations.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS access_log (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            ip_address   TEXT,
            student_id   TEXT,
            page         TEXT,
            method       TEXT,
            status_code  INTEGER,
            timestamp    REAL,
            duration_ms  REAL,
            user_agent   TEXT
        )
    """)
    conn.commit()
    conn.close()

def log_access(ip, student_id, page, method, status_code, duration_ms, user_agent):
    try:
        conn = sqlite3.connect("violations.db")
        c = conn.cursor()
        c.execute(
            "INSERT INTO access_log (ip_address,student_id,page,method,status_code,timestamp,duration_ms,user_agent) VALUES (?,?,?,?,?,?,?,?)",
            (ip, student_id, page, method, status_code, time.time(), round(duration_ms,1), user_agent[:200] if user_agent else '')
        )
        conn.commit()
        conn.close()
    except Exception:
        pass  # Never crash the app because of logging

@app.before_request
def before_request_hook():
    request._start_time = time.time()

@app.after_request
def after_request_log(response):
    # Skip static files to keep log clean
    if not request.path.startswith('/static'):
        duration_ms = (time.time() - getattr(request, '_start_time', time.time())) * 1000
        student_id  = (request.args.get('student_id') or
                       (request.form.get('student_id') if request.method == 'POST' else None) or
                       (request.json.get('student_id') if request.is_json else None) or '')
        log_access(
            ip          = request.remote_addr,
            student_id  = student_id,
            page        = request.path,
            method      = request.method,
            status_code = response.status_code,
            duration_ms = duration_ms,
            user_agent  = request.headers.get('User-Agent', '')
        )
    return response

# ===============================
# WEBCAM
# ===============================
video_capture = None
capture_lock = Lock()
monitoring_results = {'status': 'idle', 'message': '', 'warnings': []}

# ===============================
# ROUTES — Standard
# ===============================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/exam')
def exam():
    student_id = request.args.get('student_id', '')
    is_running = False
    return render_template('exam.html', duration=EXAM_DURATION,
                           student_id=student_id, is_running=is_running)

@app.route('/start_monitoring', methods=['POST'])
def start_monitoring():
    student_id   = request.form.get('student_id', '')
    model_choice = request.form.get('model_choice', 'mediapipe')

    if not student_id:
        return jsonify({'error': 'Student ID cannot be empty.'}), 400

    active_sessions[student_id] = ExamSession(student_id, model_choice)
    session_id = active_sessions[student_id].session_id

    return jsonify({
        'status':     'started',
        'message':    'Session created successfully.',
        'session_id': session_id           # Return so JS can link to activity log
    }), 200

@app.route('/process_frame', methods=['POST'])
def process_frame():
    data       = request.json
    student_id = data.get('student_id')

    if not student_id or student_id not in active_sessions:
        return jsonify({'error': 'No active session found for this student.'}), 400

    session = active_sessions[student_id]
    result  = session.process_base64_frame(data['image'])
    return jsonify(result)

# ===============================
# ROUTES — Registration  (rate-limited)
# ===============================
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        # Apply rate limit programmatically if limiter is available
        if RATE_LIMITING_ENABLED:
            try:
                limiter.limit("5 per minute")(lambda: None)()
            except Exception:
                return jsonify({'error': 'Too many registration attempts. Try again in a minute.'}), 429

        if request.is_json:
            student_id  = request.json.get('student_id')
            image_b64   = request.json.get('image_base64')
        else:
            student_id  = request.form.get('student_id')
            image_b64   = request.form.get('image_base64')

        if not image_b64 or not student_id:
            return "Missing student ID or image", 400

        import numpy as np
        import base64
        if "," in image_b64:
            image_b64 = image_b64.split(",")[1]
        np_img = np.frombuffer(base64.b64decode(image_b64), np.uint8)
        img    = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        success, msg = face_module.register_face_from_image(student_id, img)
        if success:
            return redirect(url_for('index'))
        else:
            # Return descriptive error (now includes quality reasons)
            return msg, 400
    return render_template('register.html')

# ===============================
# ROUTES — Verification  (rate-limited)
# ===============================
@app.route('/verify', methods=['GET', 'POST'])
def verify():
    student_id_get = request.args.get('student_id', '')
    if request.method == 'POST':
        if RATE_LIMITING_ENABLED:
            try:
                limiter.limit("5 per minute")(lambda: None)()
            except Exception:
                return jsonify({'error': 'Too many verification attempts. Try again in a minute.'}), 429

        if request.is_json:
            student_id = request.json.get('student_id')
            image_b64  = request.json.get('image_base64')
        else:
            student_id = request.form.get('student_id')
            image_b64  = request.form.get('image_base64')

        if not image_b64 or not student_id:
            return "Missing student ID or image", 400

        import numpy as np
        import base64
        if "," in image_b64:
            image_b64 = image_b64.split(",")[1]
        np_img = np.frombuffer(base64.b64decode(image_b64), np.uint8)
        img    = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        success, msg = face_module.verify_face_from_image(student_id, img)
        if success:
            return redirect(url_for('result',
                                    message=f"Verification successful for {student_id}! You may now proceed.",
                                    action_url=url_for('exam', student_id=student_id),
                                    action_text="Attempt Exam"))
        else:
            # Pass quality error message through to result page
            fail_msg = f"Verification Failed: {msg}"
            return redirect(url_for('result',
                                    message=fail_msg,
                                    action_url=url_for('verify'),
                                    action_text="Try Again"))
    return render_template('verify.html', student_id=student_id_get)

@app.route('/result')
def result():
    message     = request.args.get('message', 'No message provided.')
    action_url  = request.args.get('action_url', '')
    action_text = request.args.get('action_text', '')
    return render_template('result.html', message=message,
                           action_url=action_url, action_text=action_text)

# ===============================
# ROUTES — Activity Log  (NEW in Production_2)
# ===============================
@app.route('/activity_log')
def activity_log():
    """
    Dashboard showing the full exam activity timeline for a student.
    Query params:
        student_id  — required
        session_id  — optional, filters to a specific exam attempt
    """
    student_id = request.args.get('student_id', '').strip()
    session_id = request.args.get('session_id', '').strip() or None

    if not student_id:
        return render_template('activity_log.html',
                               student_id='', session_id='',
                               summary=None, events=[], error="Please provide a Student ID.")

    summary = None
    events  = []
    error   = None

    try:
        init_db()                          # Ensure tables exist
        if session_id:
            summary = get_session_summary(student_id, session_id)
            events  = summary['events'] if summary else []
        else:
            rows = get_activity_log(student_id)
            events = rows
    except Exception as e:
        error = str(e)

    return render_template('activity_log.html',
                           student_id=student_id,
                           session_id=session_id or '',
                           summary=summary,
                           events=events,
                           error=error)

@app.route('/api/activity_log')
def api_activity_log():
    """JSON API endpoint for activity log data (used by the chart in activity_log.html)."""
    student_id = request.args.get('student_id', '').strip()
    session_id = request.args.get('session_id', '').strip() or None

    if not student_id:
        return jsonify({'error': 'student_id required'}), 400

    init_db()
    if session_id:
        summary = get_session_summary(student_id, session_id)
        return jsonify(summary or {})
    else:
        rows = get_activity_log(student_id)
        return jsonify({'events': rows})


# ===============================
# STUDENT REGISTRY DASHBOARD  (NEW)
# ===============================
@app.route('/students')
def students():
    """Admin page — all registered students with their photos and registration time."""
    import glob
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    pkl_files = glob.glob(os.path.join(data_dir, '*_encoding.pkl'))

    student_list = []
    for f in sorted(pkl_files):
        sid = os.path.basename(f).replace('_encoding.pkl', '')
        # Skip debug/temp entries
        if sid.startswith('debug') or sid.startswith('test'):
            continue
        reg_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(f)))
        has_photo = os.path.exists(os.path.join(data_dir, f'{sid}_photo.jpg'))
        student_list.append({'id': sid, 'registered_at': reg_time, 'has_photo': has_photo})

    return render_template('students.html', students=student_list, total=len(student_list))

@app.route('/student_photo/<student_id>')
def student_photo(student_id):
    """Serve a student's registration photo."""
    from flask import send_file, abort
    # Sanitize — only allow alphanumeric, dash, underscore
    import re
    if not re.match(r'^[\w\-]+$', student_id):
        abort(400)
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    photo_path = os.path.join(data_dir, f'{student_id}_photo.jpg')
    if not os.path.exists(photo_path):
        abort(404)
    return send_file(photo_path, mimetype='image/jpeg')

# ===============================
# ACCESS LOG DASHBOARD  (NEW)
# ===============================
@app.route('/access_log')
def access_log_page():
    """Admin dashboard — shows every visit: IP, student, page, time, duration."""
    try:
        conn = sqlite3.connect("violations.db")
        c = conn.cursor()
        # Last 200 visits, newest first
        c.execute("""
            SELECT ip_address, student_id, page, method, status_code,
                   datetime(timestamp,'unixepoch','localtime') as dt,
                   duration_ms, user_agent
            FROM access_log
            ORDER BY timestamp DESC
            LIMIT 200
        """)
        rows = c.fetchall()

        # Summary: unique IPs, unique students, total hits
        c.execute("SELECT COUNT(DISTINCT ip_address), COUNT(DISTINCT student_id), COUNT(*) FROM access_log")
        stats = c.fetchone()
        conn.close()
        return render_template('access_log.html', rows=rows, stats=stats)
    except Exception as e:
        return f"<pre>Error: {e}</pre>", 500


if __name__ == '__main__':
    init_violations_db()
    init_access_log_db()
    init_db()
    # Hugging Face Spaces / ngrok require binding to 0.0.0.0 and port 7860
    app.run(host="0.0.0.0", port=7860, debug=False)