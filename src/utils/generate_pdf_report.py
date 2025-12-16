import os
from fpdf import FPDF
import re

class MedicalReportPDF(FPDF):
    def header(self):
        # Add a logo or header text if needed
        self.set_font('Arial', 'B', 10)
        self.cell(0, 10, 'HemaVision AI - Confidential Screening Report', 0, 1, 'R')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def sanitize_text(self, text):
        """
        Sanitizes text to be compatible with FPDF's Latin-1/CP1252 encoding.
        Replaces common Unicode characters with ASCII equivalents.
        """
        replacements = {
            '\u2018': "'",  # Left single quote
            '\u2019': "'",  # Right single quote
            '\u201c': '"',  # Left double quote
            '\u201d': '"',  # Right double quote
            '\u2013': '-',  # En dash
            '\u2014': '-',  # Em dash
            '\u2026': '...', # Ellipsis
            '\u2022': '-',  # Bullet
        }
        for char, replacement in replacements.items():
            text = text.replace(char, replacement)
            
        # Finally, encode to latin-1, replacing errors with '?'
        return text.encode('latin-1', 'replace').decode('latin-1')

    def write_markdown(self, text):
        """
        Parses simple Markdown (Headers #, Bold **) and renders it in PDF.
        """
        # Sanitize text first
        text = self.sanitize_text(text)

        # Set Standard Font
        self.set_font("Arial", size=11)
        
        # Split into lines to handle headers
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                self.ln(5) # Empty line spacing
                continue
            
            # --- 1. Handle Headers (### Title) ---
            if line.startswith('### ') or line.startswith('## '):
                clean_line = line.replace('#', '').strip()
                self.set_font("Arial", 'B', 14) # Bold & Large
                self.set_text_color(44, 62, 80) # Dark Blue-Grey
                self.cell(0, 10, clean_line, ln=True)
                self.set_text_color(0, 0, 0) # Reset to Black
                self.set_font("Arial", '', 11) # Reset to Normal
                continue
            
            # --- 2. Handle List Items (- Item) ---
            if line.startswith('- '):
                self.set_x(20) # Indent
                line = chr(149) + " " + line[2:] # Replace dash with Bullet point
            
            # --- 3. Handle Bold Text (**Text**) ---
            # We split the line by '**'. Even parts are normal, Odd parts are bold.
            # Example: "This is **bold** text" -> ["This is ", "bold", " text"]
            parts = line.split('**')
            
            for i, part in enumerate(parts):
                if i % 2 == 1: # Inside ** -> Bold
                    self.set_font("Arial", 'B', 11)
                    self.write(6, part)
                else: # Outside -> Normal
                    self.set_font("Arial", '', 11)
                    self.write(6, part)
            
            self.ln(6) # Move to next line after printing text

def generate_pdf_report(filename, prediction, report_text, image_path=None, heatmap_path=None, patient_info=None):
    pdf = MedicalReportPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # --- Title Section ---
    pdf.set_font("Arial", 'B', 20)
    pdf.set_text_color(41, 128, 185) # Professional Blue
    pdf.cell(0, 15, "Medical Analysis Report", ln=True, align='C')
    pdf.ln(5)

    # --- Patient Info Panel ---
    if patient_info:
        pdf.set_fill_color(245, 247, 250)
        pdf.rect(10, pdf.get_y(), 190, 35, 'F')
        
        pdf.set_font("Arial", 'B', 14)
        pdf.set_text_color(44, 62, 80)
        pdf.cell(0, 10, " Patient Information", 0, 1)
        
        pdf.set_font("Arial", 'B', 11)
        pdf.set_text_color(100, 100, 100)
        
        start_y = pdf.get_y()
        
        # Column 1
        pdf.text(15, start_y + 8, f"Name: {patient_info.get('patient_name', 'N/A')}")
        pdf.text(15, start_y + 16, f"ID: {patient_info.get('patient_id', 'N/A')}")
        
        # Column 2
        pdf.text(110, start_y + 8, f"Sex: {patient_info.get('patient_sex', 'N/A')}")
        pdf.text(110, start_y + 16, f"Age: {patient_info.get('patient_age', 'N/A')}")
        
        pdf.ln(25)

    # --- File Info ---
    pdf.set_font("Arial", 'B', 12)
    pdf.set_text_color(50, 50, 50)
    pdf.cell(40, 10, "Analyzed File:", 0, 0)
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, filename, 0, 1)
    pdf.ln(5)

    # --- NEW PAGE: Visual Evidence ---
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.set_text_color(41, 128, 185)
    pdf.cell(0, 10, "Visual Evidence", 0, 1, 'L')
    pdf.ln(5)

    if image_path and os.path.exists(image_path):
        start_y = pdf.get_y()
        
        # Log image details for debugging
        try:
            img_size = os.path.getsize(image_path)
            print(f"DEBUG: Adding image to PDF. Path: {image_path}, Size: {img_size} bytes")
        except Exception as e:
            print(f"DEBUG: Error getting image info: {e}")

        # Original
        pdf.set_font("Arial", 'B', 12)
        pdf.set_text_color(0, 0, 0)
        pdf.text(10, start_y + 5, "Original X-Ray")
        try:
            # Scale image to fit half page width roughly
            pdf.image(image_path, x=10, y=start_y + 10, w=90)
        except Exception as e:
            print(f"Error adding image: {e}")
            
        # Heatmap
        if heatmap_path and os.path.exists(heatmap_path):
            pdf.text(110, start_y + 5, "AI Heatmap Analysis")
            try:
                pdf.image(heatmap_path, x=110, y=start_y + 10, w=90)
            except Exception as e:
                print(f"Error adding heatmap: {e}")
        
        # Move cursor down below images (approx 90mm height + spacing)
        pdf.set_y(start_y + 110)
    
    elif heatmap_path and os.path.exists(heatmap_path):
        pdf.cell(200, 10, txt="Grad-CAM Heatmap:", ln=True)
        pdf.image(heatmap_path, x=10, y=pdf.get_y(), w=90)
        pdf.ln(95)

    # --- NEW PAGE: Analysis Results ---
    pdf.add_page()
    
    # --- Prediction Table ---
    pdf.set_font("Arial", 'B', 14)
    pdf.set_text_color(44, 62, 80)
    pdf.set_fill_color(240, 240, 240) # Light Gray Background
    pdf.cell(0, 10, " AI Prediction Summary", 0, 1, 'L', 1)
    pdf.ln(5)

    # Table Header
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(95, 10, "Condition", 1, 0, 'C', 1)
    pdf.cell(95, 10, "Confidence", 1, 1, 'C', 1)

    # Table Rows
    pdf.set_font("Arial", '', 11)
    # Handle prediction format: list of tuples (label, prob_str, raw_prob)
    for item in prediction:
        # Unpack safely
        if len(item) == 3:
            label, prob_str, raw_val = item
        else:
            label, prob_str = item
            raw_val = 0.0 # Default if not provided

        # Highlight high risks in Red
        if raw_val > 0.2: # Threshold for visual highlight
            pdf.set_text_color(192, 57, 43) # Red
            pdf.set_font("Arial", 'B', 11)
        else:
            pdf.set_text_color(0, 0, 0)
            pdf.set_font("Arial", '', 11)
            
        pdf.cell(95, 10, label, 1)
        pdf.cell(95, 10, prob_str, 1, ln=True)

    pdf.set_text_color(0, 0, 0) # Reset color
    pdf.ln(10)

    # --- Detailed AI Report (Using the Markdown Parser) ---
    pdf.set_font("Arial", 'B', 14)
    pdf.set_fill_color(240, 240, 240)
    pdf.set_text_color(44, 62, 80)
    pdf.cell(0, 10, " Detailed Radiologist Analysis", 0, 1, 'L', 1)
    pdf.ln(5)
    
    # Use the custom parser here!
    pdf.write_markdown(report_text)

    # --- Disclaimer ---
    pdf.ln(15)
    pdf.set_draw_color(200, 200, 200)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(2)
    pdf.set_font("Arial", 'I', 9)
    pdf.set_text_color(100, 100, 100)
    pdf.multi_cell(0, 5, "DISCLAIMER: This report is generated by HemaVision AI for screening purposes only. It is NOT a definitive medical diagnosis. Please consult a qualified specialist for interpretation and confirmation.")

    # --- Save ---
    # Use absolute path resolution
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # src/med_image
    output_folder = os.path.join(base_dir, 'app', 'static', 'reports')
        
    os.makedirs(output_folder, exist_ok=True)

    base_name = os.path.splitext(filename)[0]
    safe_name = "".join([c for c in base_name if c.isalnum() or c in ('-', '_')]).strip()
    pdf_filename = f"{safe_name}_report.pdf"
    full_path = os.path.join(output_folder, pdf_filename)
    
    try:
        pdf.output(full_path)
        print(f"✅ PDF Generated: {full_path}")
        return pdf_filename
    except Exception as e:
        print(f"❌ PDF Error: {e}")
        return None