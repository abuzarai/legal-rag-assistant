import os
from langchain_community.document_loaders import PyPDFLoader

def extract_text_from_txt(txt_path):
    with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    # For .txt we treat the whole file as a single page
    return [{
        "page_content": content,
        "metadata": {
            "source": txt_path,
            "page": 1
        }
    }]

def extract_text_from_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    extracted = []
    for doc in documents:
        extracted.append({
            "page_content": doc.page_content,
            "metadata": doc.metadata
        })
    return extracted

if __name__ == "__main__":
    test_file = "./data/raw_pdfs/case-laws//1983 C L C 944.pdf.pdf"
    if os.path.exists(test_file):
        data = extract_text_from_pdf(test_file)
        print(data[0])
