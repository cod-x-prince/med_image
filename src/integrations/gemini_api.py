import os
import json
import google.generativeai as genai
from dotenv import load_dotenv
import logging

# Configure logging
logger = logging.getLogger(__name__)

# 1. Load Environment Variables from .env
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

# Check if key was loaded successfully
if not api_key:
    logger.error("❌ CRITICAL ERROR: 'GEMINI_API_KEY' not found. Check your .env file.")
else:
    # Configure Gemini
    genai.configure(api_key=api_key)

# 2. Load Disease Info (Grounding Data)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(BASE_DIR, 'disease_info.json')

DISEASE_DB = {}
try:
    if os.path.exists(JSON_PATH):
        with open(JSON_PATH, 'r', encoding='utf-8') as f:
            DISEASE_DB = json.load(f)
    else:
        logger.warning(f"⚠️ Warning: Could not find disease_info.json at {JSON_PATH}")

except Exception as e:
    logger.error(f"Error loading disease info: {e}")

def generate_medical_report(disease_name, confidence, heatmap_context="unspecified region"):
    """
    Generates a professional medical report using Gemini.
    """
    if not api_key:
        return "AI Report Unavailable (Missing API Key). Please check server logs."

    try:
        # Initialize the model
        model = genai.GenerativeModel("gemini-2.5-flash")
        
        # Fetch medical facts
        info = DISEASE_DB.get(disease_name, {})
        
        # Construct the Prompt
        prompt = f"""
        ROLE: You are HemaVision AI, an advanced Medical Radiologist Assistant.
        TASK: Write a clear, patient-friendly screening report.
        
        FINDINGS:
        - Detected Condition: {disease_name}
        - AI Confidence: {confidence:.1f}% (Threshold met)
        - Visual Focus: {heatmap_context}
        
        MEDICAL KNOWLEDGE (For Context):
        - Description: {info.get('description', f'A condition affecting the {info.get("affected_regions", "lungs")}')}
        - Standard Treatments: {', '.join(info.get('medications', ['Consult specialist']))}
        - Curability: {info.get('curable', 'Varies')}

        REPORT STRUCTURE (Markdown):
        1. **Summary of Detection:** - State that the AI has identified signs of {disease_name} with {confidence:.1f}% confidence.
           - If confidence is < 30%, note that it is a "Low Confidence Alert" requiring verification.
           - If confidence is > 30%, note that it is a "Significant Finding".
           
        2. **Understanding {disease_name}:** - Briefly explain what this is in simple terms.
           
        3. **Clinical Recommendation:** - Suggest seeing a doctor for a follow-up (CT Scan or Physical Exam).
           - Mention that this is a screening tool, NOT a final diagnosis.
        
        Keep the tone professional, empathetic, and concise.
        """

        # Generate Response
        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        error_msg = f"Error generating AI report: {str(e)}"
        logger.error(f"❌ {error_msg}") 
        return "AI Report Unavailable. Please refer to the Prediction Summary above."