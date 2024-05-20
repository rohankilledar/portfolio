from pdf2image import convert_from_path
import pytesseract
import cv2
import numpy as np

class OCRHelper:
    def __init__(self):
        pass

    def extract_images_from_pdf(self, pdf_path):
        """
        Convert PDF pages to images.
        :param pdf_path: Path to the PDF file.
        :return: List of images.
        """
        return convert_from_path(pdf_path)

    def ocr_text_from_images(self, images):
        """
        Extract text from images using Tesseract OCR.
        :param images: List of images.
        :return: Extracted text.
        """
        extracted_text = ""
        page_no = 1
        for image in images:
            open_cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            # Convert to grayscale
            gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
            # Use adaptive thresholding to handle varying lighting conditions
            binary_image = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 8)
            # Invert the image if necessary to make text black and background white
            if np.mean(binary_image) < 127:  # If the mean intensity is higher, text is likely white
                binary_image = cv2.bitwise_not(binary_image)
            text = pytesseract.image_to_string(binary_image, config= r'--psm 6')
            extracted_text +=f"Page {page_no}:\n" + text + "\n"
            page_no += 1
        return extracted_text

    def extract_text_from_pdf(self, pdf_path):
        """
        Extract text from a PDF file using OCR.
        :param pdf_path: Path to the PDF file.
        :return: Extracted text.
        """
        images = self.extract_images_from_pdf(pdf_path)
        extracted_text = self.ocr_text_from_images(images)
        return extracted_text

if __name__ == "__main__":
    ocr_helper = OCRHelper()
    # extracted_text = ocr_helper.extract_text_from_pdf("Take Home Test/Treadclimber manual.pdf")
    extracted_text = ocr_helper.extract_text_from_pdf("Take Home Test/fan manual.pdf")
    # extracted_text = ocr_helper.ocr_text_from_images("/Users/rohankilledar/Documents/projects/pdf-extractor/output_folder/enhanced_image_1.png")
    with open("ocr_output.txt", "w") as f:
        f.write(extracted_text)
    print(extracted_text)

    print(ocr_helper.extract_images_from_pdf("troubleshooting.pdf"))