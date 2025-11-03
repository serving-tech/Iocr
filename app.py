from flask import Flask, request, jsonify
import cv2
import numpy as np
import pytesseract
import re
from datetime import datetime
import tempfile
import os

app = Flask(__name__)

class MRZParser:
    def __init__(self):
        self.mrz_pattern_td1 = re.compile(r'^[A-Z0-9<]{30}$')
        self.mrz_pattern_td2 = re.compile(r'^[A-Z0-9<]{36}$')
        self.mrz_pattern_td3 = re.compile(r'^[A-Z0-9<]{44}$')
    
    def detect_mrz_region(self, image):
        """Detect MRZ region in the image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations to detect text regions
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours for MRZ characteristics
        mrz_candidates = []
        h, w = image.shape[:2]
        
        for contour in contours:
            x, y, cw, ch = cv2.boundingRect(contour)
            aspect_ratio = cw / ch if ch > 0 else 0
            
            # MRZ is typically wide and in lower part of image
            if aspect_ratio > 5 and ch < h * 0.2 and y > h * 0.5:
                mrz_candidates.append((x, y, cw, ch))
        
        if not mrz_candidates:
            # Fallback: return bottom portion of image
            return gray[int(h*0.7):h, 0:w]
        
        # Get the largest candidate
        mrz_candidates.sort(key=lambda r: r[2] * r[3], reverse=True)
        x, y, cw, ch = mrz_candidates[0]
        
        # Add padding
        padding = 10
        y1 = max(0, y - padding)
        y2 = min(h, y + ch + padding)
        x1 = max(0, x - padding)
        x2 = min(w, x + cw + padding)
        
        return gray[y1:y2, x1:x2]
    
    def preprocess_mrz(self, mrz_region):
        """Preprocess MRZ region for better OCR"""
        # Resize for better OCR
        scale = 2
        resized = cv2.resize(mrz_region, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
        
        return denoised
    
    def extract_mrz_text(self, preprocessed):
        """Extract text from preprocessed MRZ region"""
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<'
        text = pytesseract.image_to_string(preprocessed, config=custom_config)
        
        # Clean and split into lines
        lines = [line.strip().replace(' ', '').replace('«', '<').replace('»', '<') 
                 for line in text.split('\n') if line.strip()]
        
        return lines
    
    def identify_mrz_type(self, lines):
        """Identify the type of MRZ document"""
        for line in lines:
            if self.mrz_pattern_td1.match(line):
                return 'TD1', 30  # ID cards (3 lines of 30 chars)
            elif self.mrz_pattern_td2.match(line):
                return 'TD2', 36  # Official travel documents (2 lines of 36 chars)
            elif self.mrz_pattern_td3.match(line):
                return 'TD3', 44  # Passports (2 lines of 44 chars)
        return None, None
    
    def parse_td1(self, lines):
        """Parse TD1 format (3 lines of 30 characters)"""
        if len(lines) < 3:
            return None
        
        line1, line2, line3 = lines[:3]
        
        return {
            'document_type': line1[0:2].replace('<', ''),
            'issuing_country': line1[2:5].replace('<', ''),
            'document_number': line1[5:14].replace('<', ''),
            'birth_date': self.parse_date(line2[0:6]),
            'sex': line2[7],
            'expiry_date': self.parse_date(line2[8:14]),
            'nationality': line2[15:18].replace('<', ''),
            'names': self.parse_names(line3[0:30]),
            'mrz_type': 'TD1'
        }
    
    def parse_td2(self, lines):
        """Parse TD2 format (2 lines of 36 characters)"""
        if len(lines) < 2:
            return None
        
        line1, line2 = lines[:2]
        
        return {
            'document_type': line1[0:2].replace('<', ''),
            'issuing_country': line1[2:5].replace('<', ''),
            'names': self.parse_names(line1[5:36]),
            'document_number': line2[0:9].replace('<', ''),
            'nationality': line2[10:13].replace('<', ''),
            'birth_date': self.parse_date(line2[13:19]),
            'sex': line2[20],
            'expiry_date': self.parse_date(line2[21:27]),
            'mrz_type': 'TD2'
        }
    
    def parse_td3(self, lines):
        """Parse TD3 format (2 lines of 44 characters)"""
        if len(lines) < 2:
            return None
        
        line1, line2 = lines[:2]
        
        return {
            'document_type': line1[0:2].replace('<', ''),
            'issuing_country': line1[2:5].replace('<', ''),
            'names': self.parse_names(line1[5:44]),
            'document_number': line2[0:9].replace('<', ''),
            'nationality': line2[10:13].replace('<', ''),
            'birth_date': self.parse_date(line2[13:19]),
            'sex': line2[20],
            'expiry_date': self.parse_date(line2[21:27]),
            'mrz_type': 'TD3'
        }
    
    def parse_names(self, name_field):
        """Parse surname and given names from MRZ field"""
        names = name_field.replace('<', ' ').strip().split('  ')
        return {
            'surname': names[0].strip() if len(names) > 0 else '',
            'given_names': names[1].strip() if len(names) > 1 else ''
        }
    
    def parse_date(self, date_str):
        """Parse date from YYMMDD format"""
        try:
            year = int(date_str[0:2])
            month = int(date_str[2:4])
            day = int(date_str[4:6])
            
            # Adjust year (assume < 30 is 2000s, >= 30 is 1900s)
            year = 2000 + year if year < 30 else 1900 + year
            
            return f"{year:04d}-{month:02d}-{day:02d}"
        except:
            return date_str
    
    def process_image(self, image):
        """Main processing pipeline"""
        # Detect MRZ region
        mrz_region = self.detect_mrz_region(image)
        
        # Preprocess
        preprocessed = self.preprocess_mrz(mrz_region)
        
        # Extract text
        lines = self.extract_mrz_text(preprocessed)
        
        # Identify MRZ type
        mrz_type, line_length = self.identify_mrz_type(lines)
        
        if not mrz_type:
            return None
        
        # Parse based on type
        if mrz_type == 'TD1':
            return self.parse_td1(lines)
        elif mrz_type == 'TD2':
            return self.parse_td2(lines)
        elif mrz_type == 'TD3':
            return self.parse_td3(lines)
        
        return None


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'}), 200


@app.route('/parse-mrz', methods=['POST'])
def parse_mrz():
    """Parse MRZ from uploaded image"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    
    try:
        # Read image
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image file'}), 400
        
        # Process image
        parser = MRZParser()
        result = parser.process_image(image)
        
        if result is None:
            return jsonify({'error': 'Could not detect or parse MRZ'}), 404
        
        return jsonify({
            'success': True,
            'data': result
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'Processing error: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)