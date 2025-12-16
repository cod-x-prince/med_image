import pydicom
import numpy as np
from PIL import Image
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_dicom(dicom_path, output_folder):
    """
    Reads a DICOM file, normalizes it, and saves it as a standard PNG 
    for the AI model to use. Returns the new image path and metadata.
    """
    try:
        logger.info(f"🔄 Processing DICOM file: {dicom_path}")
        
        # 1. Read the raw DICOM file
        if not os.path.exists(dicom_path):
            logger.error(f"❌ DICOM file not found: {dicom_path}")
            return None, None
            
        dcm = pydicom.dcmread(dicom_path)
        logger.info(f"✅ DICOM loaded. Shape: {dcm.pixel_array.shape}, Type: {dcm.pixel_array.dtype}")
        
        pixel_array = dcm.pixel_array

        # 2. Handle Photometric Interpretation (Invert if needed)
        photometric = getattr(dcm, 'PhotometricInterpretation', 'MONOCHROME2')
        logger.info(f"📊 Photometric Interpretation: {photometric}")
        
        if photometric == "MONOCHROME1":
            logger.info("🔄 Inverting MONOCHROME1 to MONOCHROME2")
            pixel_array = np.max(pixel_array) - pixel_array

        # 3. Intelligent Normalization (16-bit -> 8-bit)
        logger.info(f"📈 Normalizing data range: {pixel_array.min()} to {pixel_array.max()}")
        
        pixel_array = pixel_array.astype(float)
        
        # Handle potential division by zero
        if pixel_array.max() == 0:
            logger.warning("⚠️ All pixel values are zero - using fallback normalization")
            pixel_array = np.zeros_like(pixel_array)
        else:
            pixel_array = (np.maximum(pixel_array, 0) / pixel_array.max()) * 255.0
        
        pixel_array = np.uint8(pixel_array)
        logger.info(f"✅ Normalized to range: {pixel_array.min()} to {pixel_array.max()}")

        # 4. Save as PNG for the AI
        image = Image.fromarray(pixel_array)
        base_filename = os.path.splitext(os.path.basename(dicom_path))[0]
        output_filename = f"{base_filename}_converted.png"
        output_path = os.path.join(output_folder, output_filename)
        
        # Ensure output directory exists
        os.makedirs(output_folder, exist_ok=True)
        
        image.save(output_path)
        logger.info(f"💾 Saved converted image: {output_path}")

        # 5. Extract Metadata (Useful for the Doctor's Report)
        metadata = {
            "Patient ID": str(dcm.get("PatientID", "Unknown")),
            "Sex": str(dcm.get("PatientSex", "Unknown")),
            "Age": str(dcm.get("PatientAge", "Unknown")),
            "Modality": str(dcm.get("Modality", "XR")),
            "Body Part": str(dcm.get("BodyPartExamined", "Chest")),
            "View Position": str(dcm.get("ViewPosition", "PA")),
            "Image Size": f"{dcm.pixel_array.shape[1]}x{dcm.pixel_array.shape[0]}",
            "Bits Stored": str(dcm.get("BitsStored", "Unknown"))
        }
        
        logger.info(f"📋 Extracted metadata: {metadata}")

        return output_path, metadata

    except Exception as e:
        logger.error(f"❌ DICOM Processing Error: {str(e)}", exc_info=True)
        return None, None

# Debug function to test DICOM processing
def debug_dicom_file(dicom_path):
    """Test function to debug DICOM files without full processing"""
    try:
        dcm = pydicom.dcmread(dicom_path)
        print("=== DICOM DEBUG INFO ===")
        print(f"File: {dicom_path}")
        print(f"Shape: {dcm.pixel_array.shape}")
        print(f"Data type: {dcm.pixel_array.dtype}")
        print(f"Value range: {dcm.pixel_array.min()} to {dcm.pixel_array.max()}")
        print(f"Photometric Interpretation: {getattr(dcm, 'PhotometricInterpretation', 'Not found')}")
        print("Available tags:")
        for tag in dcm.dir():
            if hasattr(dcm, tag):
                value = getattr(dcm, tag)
                if isinstance(value, (str, int, float)):
                    print(f"  {tag}: {value}")
        print("========================")
        return True
    except Exception as e:
        print(f"❌ DICOM Debug Error: {e}")
        return False