from flask import Flask, request, jsonify
import cv2
import numpy as np
import pytesseract
import re
from datetime import datetime
import os
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Debug: Save intermediate images
DEBUG_SAVE = True
DEBUG_DIR = "/tmp/mrz_debug"
os.makedirs(DEBUG_DIR, exist_ok=True)


class MRZParser:
    def __init__(self):
        self.mrz_patterns = {
            'TD1': (re.compile(r'^[A-Z0-9<]{30}$'), 30, 3),
            'TD2': (re.compile(r'^[A-Z0-9<]{36}$'), 36, 2),
            'TD3': (re.compile(r'^[A-Z0-9<]{44}$'), 44, 2)
        }

    def clean_line(self, line):
        """Aggressive cleaning of OCR output"""
        line = line.upper()
        # Replace common OCR errors
        line = line.replace('O', '0').replace('I', '1').replace('B', '8')
        line = line.replace('S', '5').replace('Z', '2').replace('G', '6')
        line = line.replace('«', '<').replace('»', '<').replace('>', '<')
        line = line.replace('|', '1').replace('!', '1').replace(' ', '')
        # Keep only valid MRZ chars
        line = re.sub(r'[^A-Z0-9<]', '', line)
        return line

    def detect_mrz_region(self, image):
        """Detect MRZ using multiple strategies"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Strategy 1: Morphological closing for text blocks
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            aspect = cw / float(ch) if ch > 0 else 0
            area = cw * ch
            if (aspect > 8 and  # Very wide
                ch < h * 0.15 and
                y > h * 0.5 and
                area > 5000):
                candidates.append((x, y, cw, ch))

        if candidates:
            # Largest by area
            candidates.sort(key=lambda x: x[2] * x[3], reverse=True)
            x, y, cw, ch = candidates[0]
            pad = 15
            return gray[max(0, y - pad):min(h, y + ch + pad),
                        max(0, x - pad):min(w, x + cw + pad)]

        # Strategy 2: Bottom 30% fallback
        logger.info("Using bottom region fallback")
        return gray[int(h * 0.7):h, 0:w]

    def preprocess_mrz(self, region):
        """Enhanced preprocessing"""
        # Resize
        scale = 3.0
        h, w = region.shape
        resized = cv2.resize(region, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

        # Multiple thresholding
        thresh1 = cv2.adaptiveThreshold(resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
        thresh2 = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        thresh = cv2.bitwise_or(thresh1, thresh2)

        # Denoise
        denoised = cv2.fastNlMeansDenoising(thresh, None, h=15, templateWindowSize=7, searchWindowSize=21)

        # Slight dilation to connect broken chars
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        final = cv2.dilate(denoised, kernel, iterations=1)

        return final

    def extract_mrz_lines(self, preprocessed):
        """Extract and normalize MRZ lines"""
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<'
        text = pytesseract.image_to_string(preprocessed, config=custom_config)

        raw_lines = [self.clean_line(line) for line in text.split('\n') if line.strip() and len(self.clean_line(line)) > 20]

        # Group by approximate length
        td1_lines = [l.ljust(30, '<')[:30] for l in raw_lines if 25 <= len(l) <= 35]
        td2_lines = [l.ljust(36, '<')[:36] for l in raw_lines if 32 <= len(l) <= 40]
        td3_lines = [l.ljust(44, '<')[:44] for l in raw_lines if 38 <= len(l) <= 48]

        return td1_lines, td2_lines, td3_lines

    def identify_mrz_type(self, td1_lines, td2_lines, td3_lines):
        """Identify MRZ type based on line count and pattern"""
        if len(td1_lines) >= 3:
            return 'TD1', td1_lines[:3]
        if len(td3_lines) >= 2:
            return 'TD3', td3_lines[:2]
        if len(td2_lines) >= 2:
            return 'TD2', td2_lines[:2]
        return None, None

    def validate_check_digit(self, field, check_digit_char):
        """Validate MRZ check digit"""
        if check_digit_char == '<':
            return True
        weights = [7, 3, 1]
        total = 0
        for i, c in enumerate(field):
            if c == '<':
                val = 0
            elif c.isdigit():
                val = int(c)
            else:
                val = ord(c) - ord('A') + 10
            total += val * weights[i % 3]
        return (total % 10) == int(check_digit_char)

    def parse_date(self, date_str):
        """Parse YYMMDD → YYYY-MM-DD"""
        try:
            if len(date_str) != 6 or not date_str.isdigit():
                return None
            yy, mm, dd = int(date_str[0:2]), int(date_str[2:4]), int(date_str[4:6])
            if not (1 <= mm <= 12 and 1 <= dd <= 31):
                return None
            year = 2000 + yy if yy < 30 else 1900 + yy
            return f"{year:04d}-{mm:02d}-{dd:02d}"
        except:
            return None

    def parse_names(self, field):
        parts = [p.strip() for p in field.replace('<', ' ').split('  ') if p.strip()]
        surname = parts[0] if parts else ''
        given = ' '.join(parts[1:]) if len(parts) > 1 else ''
        return {'surname': surname, 'given_names': given}

    def parse_td1(self, lines):
        l1, l2, l3 = lines
        data = {
            'document_type': l1[0:2].replace('<', ''),
            'issuing_country': l1[2:5],
            'document_number': l1[5:14].replace('<', ''),
            'check_document_number': l1[14],
            'birth_date': self.parse_date(l2[0:6]),
            'check_birth_date': l2[6],
            'sex': l2[7] if l2[7] in 'MF<' else '',
            'expiry_date': self.parse_date(l2[8:14]),
            'check_expiry_date': l2[14],
            'nationality': l2[15:18],
            'optional1': l2[18:29].replace('<', ''),
            'check_composite': l2[29],
            'names': self.parse_names(l3),
            'mrz_type': 'TD1'
        }
        # Validate check digits
        valid = (
            self.validate_check_digit(l1[5:14], l1[14]) and
            self.validate_check_digit(l2[0:6], l2[6]) and
            self.validate_check_digit(l2[8:14], l2[14]) and
            self.validate_check_digit(l2[0:29], l2[29])
        )
        data['check_digits_valid'] = valid
        return data

    def parse_td2_td3(self, lines, mrz_type):
        l1, l2 = lines
        length = 44 if mrz_type == 'TD3' else 36
        data = {
            'document_type': l1[0:2].replace('<', ''),
            'issuing_country': l1[2:5],
            'names': self.parse_names(l1[5:length]),
            'document_number': l2[0:9].replace('<', ''),
            'check_document_number': l2[9],
            'nationality': l2[10:13],
            'birth_date': self.parse_date(l2[13:19]),
            'check_birth_date': l2[19],
            'sex': l2[20] if l2[20] in 'MF<' else '',
            'expiry_date': self.parse_date(l2[21:27]),
            'check_expiry_date': l2[27],
            'optional': l2[28:length].replace('<', ''),
            'check_composite': l2[length-1] if mrz_type == 'TD3' else None,
            'mrz_type': mrz_type
        }
        # Validate
        valid = (
            self.validate_check_digit(l2[0:9], l2[9]) and
            self.validate_check_digit(l2[13:19], l2[19]) and
            self.validate_check_digit(l2[21:27], l2[27])
        )
        if mrz_type == 'TD3':
            valid = valid and self.validate_check_digit(l2[0:36] + l1[0:9], l2[43])
        data['check_digits_valid'] = valid
        return data

    def process_image(self, image):
        """Main pipeline"""
        try:
            # 1. Detect region
            region = self.detect_mrz_region(image)
            if DEBUG_SAVE:
                cv2.imwrite(os.path.join(DEBUG_DIR, "01_region.jpg"), region)

            # 2. Preprocess
            preprocessed = self.preprocess_mrz(region)
            if DEBUG_SAVE:
                cv2.imwrite(os.path.join(DEBUG_DIR, "02_preprocessed.jpg"), preprocessed)

            # 3. Extract lines
            td1_lines, td2_lines, td3_lines = self.extract_mrz_lines(preprocessed)

            # 4. Identify type
            mrz_type, lines = self.identify_mrz_type(td1_lines, td2_lines, td3_lines)
            if not mrz_type or len(lines) == 0:
                logger.warning("MRZ type not identified")
                return None

            # 5. Parse
            if mrz_type == 'TD1':
                result = self.parse_td1(lines)
            else:
                result = self.parse_td2_td3(lines, mrz_type)

            logger.info(f"MRZ parsed: {mrz_type}")
            return result

        except Exception as e:
            logger.error(f"Processing error: {str(e)}")
            return None


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'}), 200


@app.route('/parse-mrz', methods=['POST'])
def parse_mrz():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    try:
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image is None:
            return jsonify({'error': 'Invalid image format'}), 400

        parser = MRZParser()
        result = parser.process_image(image)

        if result is None:
            return jsonify({'error': 'Could not detect or parse MRZ'}), 404

        return jsonify({
            'success': True,
            'data': result
        }), 200

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({'error': 'Server error'}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)