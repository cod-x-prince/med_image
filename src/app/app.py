import os
import json
import logging
from flask import Flask, render_template, request, jsonify, url_for, redirect, flash, send_from_directory, current_app
from werkzeug.utils import secure_filename
from flask_login import login_required
from celery.result import AsyncResult
from .celery_utils import make_celery

from med_image.modules.safe_range import safe_ranges
from .db import init_app
from .models import db, Upload, User
from med_image.utils.clinical_metrics import ClinicalMetrics
import torch
from datetime import datetime
import cv2
import traceback
import sys
from med_image.config import Config
from med_image.inference.inference import load_model

logger = logging.getLogger(__name__)

# Adjust path to import from parent directory if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'dcm'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    # Ensure consistent database path - REMOVED override to respect Config
    # basedir = os.path.abspath(os.path.dirname(__file__))
    # app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{os.path.join(basedir, "uploads.db")}'
    
    config_class.init_app(app)

    # Initialize Database
    init_app(app)

    # Initialize Auth (Import here to avoid circular dependency)
    from .auth import auth_bp, login_manager
    login_manager.init_app(app)
    login_manager.login_view = 'auth.login'
    app.register_blueprint(auth_bp)
    
    # Register Middleware (Import here to avoid circular dependency)
    from .middleware import register_middleware
    register_middleware(app)

    # Initialize Celery
    celery = make_celery(app)
    app.celery = celery
    
    # Import tasks AFTER Celery is initialized and set as default
    # This ensures @shared_task uses the configured app (eager mode)
    from .tasks import analyze_image_task

    # Ensure directories exist
    print(f"📂 Configured UPLOAD_FOLDER: {app.config['UPLOAD_FOLDER']}")
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['HEATMAP_FOLDER'], exist_ok=True)
    os.makedirs(app.config['REPORT_FOLDER'], exist_ok=True)

    # Load Model (Lazy loading or at startup)
    try:
        from med_image.inference.model_loader import get_model, get_model_info
        
        # Initialize model
        model = get_model()
        model_info = get_model_info()
        
        logger.info(f"🚀 Model loaded: {model_info['selected_model']}")
        logger.info(f"📱 Device: {model_info['device']}")
        
    except Exception as e:
        logger.critical(f"❌ Failed to load model: {e}")
        # We might want to exit or continue with limited functionality
    
    @app.route('/')
    @login_required
    def index():
        logger.info("📢 Serving Index Page")
        return render_template('index.html')

    @app.route("/predict", methods=["POST"])
    @login_required
    def predict_image():
        logger.info("⚡ /predict route called")

        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        if not allowed_file(file.filename):
             return jsonify({"error": "Invalid file type"}), 400

        try:
            threshold = float(request.form.get("threshold", 0.15))
        except Exception:
            threshold = 0.15

        scan_type = request.form.get("scan_type", "standard") # Default to standard, UI overrides
        
        filename = secure_filename(file.filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(image_path)

        # Get Patient Details
        patient_details = {
            "patient_id": request.form.get("patient_id"),
            "patient_name": request.form.get("patient_name"),
            "patient_sex": request.form.get("patient_sex"),
            "patient_age": request.form.get("patient_age")
        }

        # Trigger Async Task
        task = analyze_image_task.delay(filename, threshold, patient_details, scan_type)

        return jsonify({
            "task_id": task.id,
            "status_url": url_for('task_status', task_id=task.id),
            "message": f"Analysis started ({scan_type} mode)."
        }), 202

    @app.route('/rescan/<filename>')
    @login_required
    def rescan(filename):
        """Re-analyze an existing image with different scan type"""
        scan_type = request.args.get('scan_type', 'ensemble')
        threshold = request.args.get('threshold', 0.15)
        
        # Get existing upload record
        upload = Upload.query.filter_by(filename=filename).first()
        if not upload:
            flash("File not found in history", "danger")
            return redirect(url_for('history'))
        
        # Create new analysis task with different scan type
        patient_details = {
            "patient_id": upload.patient_id,
            "patient_name": upload.patient_name,
            "patient_sex": upload.patient_sex,
            "patient_age": upload.patient_age
        }
        
        task = analyze_image_task.delay(filename, threshold, patient_details, scan_type)
        
        flash(f"Started {scan_type} analysis on existing image...", "info")
        # Redirect to a loading page or tasks status? 
        # For simplicity, redirect to history or stay on result with a loading indicator?
        # The user's snippet suggested: return redirect(url_for('task_status', task_id=task.id))
        # But task_status is a JSON API. We should probably redirect to a "processing" view or index with task ID.
        # Let's redirect to index with a special flag/modal? 
        # Actually, best UX is probably to go to history or reload result page with a "processing" state?
        # The user's original request code had: return redirect(url_for('task_status', task_id=task.id)) which is weird if it returns JSON.
        # Let's assume the frontend handles the task_id.
        # We'll redirect to the result page but maybe with a task_id parameter to trigger poll?
        # Or just redirect to history_page.
        return redirect(url_for('history')) 

    @app.route('/task-status/<task_id>')
    @login_required
    def task_status(task_id):
        task = AsyncResult(task_id, app=app.celery)
        if task.state == 'PENDING':
            response = {'state': task.state, 'status': 'Pending...'}
        elif task.state != 'FAILURE':
            # Parse result if it's a JSON string (which it is from our task)
            result = task.result
            if isinstance(result, str):
                try:
                    result = json.loads(result)
                except:
                    pass
                    
            response = {
                'state': task.state,
                'result': result
            }
        else:
            response = {'state': task.state, 'status': str(task.info)}
        return jsonify(response)

    @app.route('/result/<filename>')
    @login_required
    def view_result(filename):
        # Fetch from DB to verify existence and get report/prediction
        row = Upload.query.filter_by(filename=filename).first()
        
        if not row:
            flash("Result not found.", "danger")
            return redirect(url_for('history'))
        
        if row.prediction:
            try:
                prediction = json.loads(row.prediction)
                sorted_prediction = sorted(
                    [(label, f"{prob * 100:.3f}%", prob) for label, prob in prediction.items()],
                    key=lambda x: x[2],
                    reverse=True
                )
            except:
                sorted_prediction = []
        else:
            sorted_prediction = []

        # Reconstruct paths with cache busting
        timestamp = int(datetime.now().timestamp())
        
        image_path = url_for('static', filename=f'uploads/{filename}')
        if filename.lower().endswith('.dcm'):
             png_name = filename.replace('.dcm', '.png')
             if os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], png_name)):
                 image_path = url_for('static', filename=f'uploads/{png_name}')
        
        # Add timestamp to image path
        image_path = f"{image_path}?v={timestamp}"

        # Heatmap handling (check specific scan type heatmap first, fallback to generic)
        # Note: tasks.py saves as heatmap_{scan_type}_{filename}.
        # We need to find the correct file. The DB 'scan_type' field knows what created it.
        heatmap_filename = f"heatmap_{row.scan_type}_{filename}"
        heatmap_path_disk = os.path.join(app.config['HEATMAP_FOLDER'], heatmap_filename)
        
        # Fallback to older naming convention if newer not found
        if not os.path.exists(heatmap_path_disk):
             heatmap_filename = f"heatmap_{filename}"
             heatmap_path_disk = os.path.join(app.config['HEATMAP_FOLDER'], heatmap_filename)

        heatmap_url = None
        if os.path.exists(heatmap_path_disk):
            heatmap_url = url_for('static', filename=f'heatmaps/{heatmap_filename}')
            heatmap_url = f"{heatmap_url}?v={timestamp}"

        pdf_path = url_for('download_pdf', filename=f"{os.path.splitext(filename)[0]}_report.pdf")
        if os.path.exists(os.path.join(app.config['REPORT_FOLDER'], f"{os.path.splitext(filename)[0]}_report.pdf")):
             pdf_path = f"{pdf_path}?v={timestamp}"
        else:
             pdf_path = "#"

        # Populate metadata from DB
        metadata = {
            "Patient ID": row.patient_id,
            "Patient Name": row.patient_name,
            "Sex": row.patient_sex,
            "Age": row.patient_age,
            "View Position": "PA",
            # New Metadata
            "scan_type": row.scan_type,
            "model_used": row.model_used,
            "inference_time": f"{row.inference_time:.2f}s" if row.inference_time else "N/A",
            "auc_score": row.auc_score
        }

        return render_template("result.html",
                               image_path=image_path,
                               prediction=sorted_prediction,
                               pdf_path=pdf_path,
                               heatmap_url=heatmap_url,
                               safe_ranges=safe_ranges,
                               ClinicalMetrics=ClinicalMetrics,
                               metadata=metadata,
                               filename=filename) # Pass filename for rescan links

    @app.route('/reports/<path:filename>')
    @login_required
    def download_pdf(filename):
        return send_from_directory(app.config['REPORT_FOLDER'], filename, as_attachment=True)

    @app.route('/history')
    @login_required
    def history():
        search_query = request.args.get('search')
        if search_query:
            uploads = Upload.query.filter(Upload.patient_id.ilike(f"%{search_query}%")).order_by(Upload.date.desc()).all()
        else:
            uploads = Upload.query.order_by(Upload.date.desc()).all()
        
        history_data = []
        timestamp = int(datetime.now().timestamp())
        
        for upload in uploads:
            # Check for files
            heatmap_filename = f"heatmap_{upload.filename}"
            heatmap_path = os.path.join(app.config['HEATMAP_FOLDER'], heatmap_filename)
            pdf_filename = f"{os.path.splitext(upload.filename)[0]}_report.pdf"
            pdf_path_disk = os.path.join(app.config['REPORT_FOLDER'], pdf_filename)
            
            try:
                prediction = json.loads(upload.prediction)
            except:
                prediction = {}

            history_data.append({
                "filename": upload.filename,
                "date": upload.date,
                "prediction": prediction,
                "heatmap_exists": os.path.exists(heatmap_path),
                "pdf_exists": os.path.exists(pdf_path_disk),
                "patient_id": upload.patient_id,
                "timestamp": timestamp # Pass timestamp to template
            })
        
        return render_template('history.html', history=history_data)

    @app.route('/delete_all_history', methods=['POST'])
    @login_required
    def delete_all_history():
        try:
            db.session.query(Upload).delete()
            db.session.commit()

            # Delete all files in static folders
            folders_to_clear = [
                app.config['UPLOAD_FOLDER'],
                app.config['HEATMAP_FOLDER'],
                app.config['REPORT_FOLDER']
            ]
            
            for folder in folders_to_clear:
                if os.path.exists(folder):
                    for filename in os.listdir(folder):
                        file_path = os.path.join(folder, filename)
                        try:
                            if os.path.isfile(file_path) or os.path.islink(file_path):
                                os.unlink(file_path)
                        except Exception as e:
                            logger.error(f"Failed to delete {file_path}: {e}")
            
            flash("All upload history and files deleted successfully.", "success")
        except Exception as e:
            logger.error(f"❌ Failed to delete all history: {e}")
            flash("Failed to delete all history.", "danger")
        return redirect(url_for('history'))

    @app.route('/delete_selected_history', methods=['POST'])
    @login_required
    def delete_selected_history():
        selected_files = request.form.getlist('selected_files')
        if not selected_files:
            flash("No records selected to delete.", "warning")
            return redirect(url_for('history'))

        try:
            for fname in selected_files:
                Upload.query.filter_by(filename=fname).delete()
                
                # Delete associated files
                try:
                    # Original Image
                    img_path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
                    if os.path.exists(img_path):
                        os.remove(img_path)
                        
                    # Heatmap
                    heatmap_path = os.path.join(app.config['HEATMAP_FOLDER'], f"heatmap_{fname}")
                    if os.path.exists(heatmap_path):
                        os.remove(heatmap_path)
                        
                    # PDF Report
                    base_name = os.path.splitext(fname)[0]
                    safe_name = "".join([c for c in base_name if c.isalnum() or c in ('-', '_')]).strip()
                    pdf_path = os.path.join(app.config['REPORT_FOLDER'], f"{safe_name}_report.pdf")
                    if os.path.exists(pdf_path):
                        os.remove(pdf_path)
                        
                except Exception as e:
                    logger.error(f"Failed to delete files for {fname}: {e}")

            db.session.commit()
            flash(f"Deleted {len(selected_files)} selected record(s).", "success")
        except Exception as e:
            logger.error(f"❌ Failed to delete selected history: {e}")
            flash("Failed to delete selected records.", "danger")

        return redirect(url_for('history'))

    # Debug route to test DICOM files
    @app.route("/debug-dicom", methods=["POST"])
    def debug_dicom():
        """Special route to debug DICOM file issues"""
        try:
            if 'file' not in request.files:
                return jsonify({"error": "No file provided"}), 400
                
            file = request.files['file']
            if file.filename == '':
                return jsonify({"error": "No file selected"}), 400
                
            if file and file.filename.lower().endswith('.dcm'):
                # Save temporarily
                temp_path = os.path.join(app.config['UPLOAD_FOLDER'], 'debug_temp.dcm')
                file.save(temp_path)
                
                # Debug the file
                success = debug_dicom_file(temp_path)
                
                # Clean up
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
                if success:
                    return jsonify({"message": "DICOM debug completed - check server logs"})
                else:
                    return jsonify({"error": "DICOM debug failed"}), 400
            else:
                return jsonify({"error": "Not a DICOM file"}), 400
                
        except Exception as e:
            logger.error(f"❌ DICOM debug error: {str(e)}", exc_info=True)
            return jsonify({"error": str(e)}), 500

    @app.route("/.well-known/appspecific/com.chrome.devtools.json")
    def chrome_devtools_json():
        return jsonify({})

    @app.route('/favicon.ico')
    def favicon():
        return "", 204

    return app

# For development server execution
if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)
