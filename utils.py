import os
import fitz  # PyMuPDF
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import pdfplumber
from langchain_openai import OpenAI  # Updated import statement
from langchain.prompts import PromptTemplate
import pandas as pd
import re
import io  # Import io module

# Set the TESSDATA_PREFIX environment variable to the path of the tessdata directory
os.environ['TESSDATA_PREFIX'] = r'C:\Program Files\Tesseract-OCR\tessdata'

def preprocess_image(image):
    """
    Preprocess the image to improve OCR accuracy.
    
    Args:
        image (PIL.Image): The image to preprocess.
    
    Returns:
        PIL.Image: The preprocessed image.
    """
    image = image.convert('L')  # Convert to grayscale
    image = image.filter(ImageFilter.MedianFilter())  # Apply a median filter
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2)  # Enhance contrast
    return image

def extract_ocr_text_from_image(image):
    """
    Extract OCR text from an image using Tesseract OCR.
    
    Args:
        image (PIL.Image): The image to extract OCR text from.
    
    Returns:
        str: Extracted OCR text.
    """
    preprocessed_img = preprocess_image(image)
    return pytesseract.image_to_string(preprocessed_img, config='--psm 6', lang='deu+eng')

def extract_text_and_ocr_from_pdf(pdf_content):
    """
    Extract regular text and OCR text from a PDF.
    
    Args:
        pdf_content (bytes): Byte content of the PDF file.
    
    Returns:
        tuple: Tuple containing extracted regular text and OCR text.
    """
    text = ''
    ocr_text = ''
    document = fitz.open(stream=pdf_content, filetype="pdf")
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        extracted_ocr_text = extract_ocr_text_from_image(img)
        if extracted_ocr_text:
            ocr_text += extracted_ocr_text + "\n"
        else:
            text += page.get_text() + "\n"
    return text, ocr_text

def extract_tables_from_pdf(pdf_content):
    """
    Extract tables from a PDF using PDFPlumber.
    
    Args:
        pdf_content (bytes): Byte content of the PDF file.
    
    Returns:
        list: List of dataframes, each representing a table.
    """
    tables = []
    with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
        for page in pdf.pages:
            extracted_tables = page.extract_tables()
            for table in extracted_tables:
                df = pd.DataFrame(table[1:], columns=table[0])
                tables.append(df)
    return tables

def extracted_data(pages_data):
    template = """Extract the useful information from this data, give tables in the same format as in pdf, and extracted text
     group it under entities by Extracting relationships between entities and summarizing key information from the document.
       : {pages}
"""

    prompt_template = PromptTemplate(input_variables=["pages"], template=template)
    llm = OpenAI(temperature=0.7)
    full_response = llm(prompt_template.format(pages=pages_data))
    return full_response

def create_docs(user_pdf_list):
    """
    Create documents from PDF files.
    
    Args:
        user_pdf_list (list): List of PDF files.
    
    Returns:
        pandas.DataFrame: DataFrame containing extracted data.
    """
    df = pd.DataFrame()
    
    for uploaded_file in user_pdf_list:
        pdf_content = uploaded_file.read()
        text_data, ocr_text_data = extract_text_and_ocr_from_pdf(pdf_content)
        tables_data = extract_tables_from_pdf(pdf_content)
        
        pages_data = text_data + "\n" + ocr_text_data + "\n" + "\n".join([table.to_string(index=False) for table in tables_data])
        llm_extracted_data = extracted_data(pages_data)
        
        pattern = r'{(.+)}'
        match = re.search(pattern, llm_extracted_data, re.DOTALL)
        if match:
            extracted_text = match.group(1)
            data_dict = eval('{' + extracted_text + '}')
            df = pd.concat([df, pd.DataFrame([data_dict])], ignore_index=True)
        else:
            print("No match found.")
    
    return df
