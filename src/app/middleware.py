import logging
import time
from flask import request, g
from flask_login import current_user
from .models import db, AuditLog

def register_middleware(app):
    # Setup logger
    audit_logger = logging.getLogger('audit')
    audit_logger.setLevel(logging.INFO)
    handler = logging.FileHandler('audit.log')
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    handler.setFormatter(formatter)
    audit_logger.addHandler(handler)

    @app.before_request
    def before_request():
        g.start_time = time.time()

    @app.after_request
    def after_request(response):
        if hasattr(g, 'start_time'):
            duration = time.time() - g.start_time
            user_id = current_user.id if current_user.is_authenticated else None
            user_str = str(user_id) if user_id else 'anonymous'
            ip = request.remote_addr
            method = request.method
            path = request.path
            status = response.status_code
            
            # Identify PHI Access
            phi_action = None
            if path.startswith('/predict') and method == 'POST':
                phi_action = 'UPLOAD_ANALYSIS'
            elif path.startswith('/result/'):
                phi_action = 'VIEW_RESULT'
            elif path.startswith('/reports/'):
                phi_action = 'DOWNLOAD_REPORT'
            elif path == '/history':
                phi_action = 'VIEW_HISTORY'

            # Log to File
            if not path.startswith('/static'):
                log_msg = f"User: {user_str} | IP: {ip} | {method} {path} | Status: {status} | Duration: {duration:.4f}s"
                if phi_action:
                    log_msg = f"[PHI: {phi_action}] {log_msg}"
                audit_logger.info(log_msg)

            # Log to Database (Only PHI or Critical Actions)
            if phi_action and user_id:
                try:
                    log_entry = AuditLog(
                        user_id=user_id,
                        action=phi_action,
                        details=f"Path: {path}, IP: {ip}, Status: {status}"
                    )
                    db.session.add(log_entry)
                    db.session.commit()
                except Exception as e:
                    audit_logger.error(f"Failed to write audit log to DB: {e}")

        return response
