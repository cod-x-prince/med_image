import os
import json
import logging
import time
import cv2
import torch
from datetime import datetime
from PIL import Image
from celery import shared_task
from flask import current_app
from .models import db, Upload

# Imports from our refactored modules
from med_image.inference.inference import preprocess, predict, DISEASE_LABELS
from med_image.utils.dual_model_loader import get_dual_model_manager
from med_image.modules.dicom_handler import process_dicom
from med_image.modules.generate_pdf_report import generate_pdf_report
from med_image.modules.gemini_api import generate_medical_report
from med_image.modules.safe_range import safe_ranges
from med_image.utils.gradcam import GradCAMPlusPlus, overlay_heatmap
from med_image.config import config as global_config

logger = logging.getLogger(__name__)

@shared_task(bind=True)
def analyze_image_task(self, filename, threshold=0.15, patient_details=None, scan_type='standard'):
    """
    Async task to process image with Dual-Mode support:
    1. Preprocess (handle DICOM)
    2. Predict (using DualModelManager)
    3. Generate Report
    4. Generate PDF
    5. Generate GradCAM
    6. Save to DB (with new metadata)
    """
    try:
        app_config = current_app.config
        upload_folder = app_config['UPLOAD_FOLDER']
        heatmap_folder = app_config['HEATMAP_FOLDER']
        report_folder = app_config['REPORT_FOLDER']
        
        if patient_details is None:
            patient_details = {}
            
        # Sanitize patient details to prevent XSS/Injection
        for key in patient_details:
            if isinstance(patient_details[key], str):
                patient_details[key] = "".join(c for c in patient_details[key] if c.isalnum() or c in " -_.,")

        image_path = os.path.join(upload_folder, filename)
        file_ext = os.path.splitext(filename)[1].lower()
        
        # === DICOM Handling ===
        display_path = image_path
        metadata = {}
        
        if file_ext == '.dcm':
            png_path, dicom_meta = process_dicom(image_path, upload_folder)
            if png_path and os.path.exists(png_path):
                display_path = png_path
                metadata = dicom_meta
                # Merge DICOM metadata if missing
                if not patient_details.get('patient_name'):
                    patient_details['patient_name'] = metadata.get('Patient Name', 'Unknown')
                if not patient_details.get('patient_id'):
                    patient_details['patient_id'] = metadata.get('Patient ID', 'Unknown')
                if not patient_details.get('patient_sex'):
                    patient_details['patient_sex'] = metadata.get('Patient Sex', 'Unknown')
                if not patient_details.get('patient_age'):
                    patient_details['patient_age'] = metadata.get('Patient Age', 'Unknown')
            else:
                raise Exception("DICOM conversion failed")
        
        # === Prediction with DualModelManager ===
        try:
            # Initialize Manager (singleton)
            model_manager = get_dual_model_manager(global_config)
            
            # Load Image as PIL for Manager
            pil_image = Image.open(display_path).convert('RGB')
            
            # Run Inference & Time it
            start_time = time.time()
            result = model_manager.predict(pil_image, scan_type, threshold)
            inference_time = time.time() - start_time
            
            # Extract formatted predictions for Report logic
            # convert list of dicts back to dict {label: prob} for existing report logic compatibility
            prediction_raw = {item['label']: item['confidence'] for item in result['predictions']}
            
            # Helper for display
            sorted_prediction = sorted(
                [(label, f"{prob * 100:.3f}%", prob) for label, prob in prediction_raw.items()],
                key=lambda x: x[2],
                reverse=True
            )
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise Exception(f"Prediction failed: {str(e)}")

        # === Report Generation ===
        primary_finding = None
        primary_conf = 0.0
        max_risk_ratio = 0.0
        
        for label, prob in prediction_raw.items():
            thresh = safe_ranges.get(label, {'val': 0.15})['val']
            risk_ratio = prob / thresh
            if risk_ratio > max_risk_ratio:
                max_risk_ratio = risk_ratio
                primary_finding = label
                primary_conf = prob

        report = "Report generation failed."
        try:
            if max_risk_ratio > 0.8: 
                grad_cam_desc = "area highlighted in the heatmap"
                # Add scan type context to report?
                report = generate_medical_report(primary_finding, primary_conf * 100, grad_cam_desc)
            else:
                report = ("Screening Result: Normal. \n\n"
                          "The AI analysis did not detect any disease patterns exceeding clinical sensitivity thresholds.")
        except Exception as e:
            logger.error(f"Report generation error: {e}")

        # === Grad-CAM ===
        # Note: GradCAM logic needs model access. DualModelManager hides models.
        # For ensemble, GradCAM is tricky. For now, we'll acquire the specific model for 'standard' or 'deep'.
        # For 'ensemble', we might default to 'standard' model for visual explanation.
        heatmap_url = None
        heatmap_filename = f"heatmap_{scan_type}_{filename}"
        heatmap_path = os.path.join(heatmap_folder, heatmap_filename)
        
        # Clean up existing heatmap
        if os.path.exists(heatmap_path):
            try:
                os.remove(heatmap_path)
            except Exception as e:
                logger.warning(f"Failed to delete existing heatmap: {e}")
        
        try:
            if prediction_raw:
                top_label = max(prediction_raw, key=prediction_raw.get)
                top_index = DISEASE_LABELS.index(top_label)
                
                # Get appropriate model for visualization
                # If ensemble, use standard model for heatmap as proxy
                vis_scan_type = scan_type if scan_type != 'ensemble' else 'standard'
                vis_model = model_manager.models.get(vis_scan_type)
                
                # Ensure we have a model (models loaded in manager)
                if vis_model:
                     # Need transforms too? GradCAM handles tensor input.
                     # We can reuse preprocess from inference.py OR use manager's transform
                    vis_transform = model_manager.transforms[vis_scan_type]
                    
                    # Prepare image for GradCAM
                    vis_image_tensor = vis_transform(pil_image).unsqueeze(0)
                    
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    vis_model = vis_model.to(device) # Should already be on device
                    vis_model.eval()

                    target_layer = getattr(vis_model.features, "denseblock4", None)
                    if target_layer is None:
                        target_layer = getattr(vis_model, "features", None)

                    gradcam = GradCAMPlusPlus(vis_model, target_layer)
                    cam = gradcam.generate(vis_image_tensor.to(device), class_idx=top_index)
                    gradcam.clear_hooks()

                    heatmap_img = overlay_heatmap(display_path, cam)
                    cv2.imwrite(heatmap_path, heatmap_img)
                    heatmap_url = f"static/heatmaps/{heatmap_filename}"
                else: 
                     logger.warning("No model available for GradCAM visualization")

        except Exception as e:
            logger.error(f"Grad-CAM failed: {e}")

        # === PDF Generation ===
        pdf_path = "#"
        try:
            report_image_path = display_path 
            pdf_filename = generate_pdf_report(filename, sorted_prediction, report, 
                                              image_path=report_image_path, 
                                              heatmap_path=heatmap_path,
                                              patient_info=patient_details)
            if pdf_filename:
                pdf_path = f"reports/{pdf_filename}"
        except Exception as e:
            logger.error(f"PDF Generation failed: {e}")

        # === Save to DB ===
        try:
            upload_entry = Upload(
                filename=filename,
                prediction=json.dumps(prediction_raw),
                report=report,
                date=datetime.now().strftime("%Y-%m-%d %H:%M"),
                patient_id=patient_details.get('patient_id'),
                patient_name=patient_details.get('patient_name'),
                patient_sex=patient_details.get('patient_sex'),
                patient_age=patient_details.get('patient_age'),
                # New Fields
                scan_type=scan_type,
                model_used=result['model_info']['name'],
                inference_time=inference_time,
                auc_score=result['performance']['auc']
            )
            db.session.add(upload_entry)
            db.session.commit()
        except Exception as e:
            logger.error(f"Database save failed: {e}")
            db.session.rollback()

        return json.dumps({
            "status": "success",
            "filename": filename,
            "prediction": sorted_prediction,
            "report": report,
            "pdf_path": pdf_path,
            "heatmap_url": heatmap_url,
            "image_url": f"static/uploads/{os.path.basename(display_path)}",
            "metadata": metadata,
            "patient_details": patient_details,
            "safe_ranges": safe_ranges,
            # Enhanced Return Data
            "scan_type": scan_type,
            "model_used": result['model_info']['name'],
            "inference_time": f"{inference_time:.2f}s",
            "performance_metrics": result['performance'],
            "confidence_note": result['model_info']['description']
        })

    except Exception as e:
        logger.error(f"Task failed: {e}")
        return json.dumps({
            "status": "error",
            "error": str(e)
        })
