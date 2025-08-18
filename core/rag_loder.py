# In rag_loder.py

import os
import pandas as pd
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader

def load_documents(doc_folder="document"):
    """
    Loads and cleans documents from the specified folder, supporting PDF and DOCX formats.
    This version includes detailed logging to verify which documents are loaded.
    """
    docs = []
    print("\n" + "="*50) # <-- ADDED FOR VISIBILITY
    print("🚀 --- Starting Document Loading Process --- 🚀")
    
    if not os.path.exists(doc_folder):
        print(f"🛑 ERROR: The directory '{doc_folder}' was not found.")
        return []

    # --- NEW: Print all files found in the directory ---
    files_in_folder = os.listdir(doc_folder)
    if not files_in_folder:
        print(f"🟡 WARNING: No files found in the '{doc_folder}' directory.")
        return []
    print(f"📄 Found {len(files_in_folder)} files: {files_in_folder}")
    # ---------------------------------------------------

    for file in files_in_folder:
        path = os.path.join(doc_folder, file)
        
        try:
            loaded_docs = []
            
            if file.endswith(".pdf"):
                loader = PyPDFLoader(path)
                loaded_docs = loader.load()

            elif file.endswith(".docx"):
                loader = UnstructuredWordDocumentLoader(path)
                loaded_docs = loader.load()

            clean_docs_from_file = [doc for doc in loaded_docs if doc.page_content and doc.page_content.strip()]
            
            if clean_docs_from_file:
                docs.extend(clean_docs_from_file)
                # This line confirms each successful load
                print(f"✅ Successfully loaded and cleaned content from: {file}")
            elif loaded_docs:
                 print(f"⚠️ Warning: No valid text content found in {file}. Skipping.")

        except Exception as e:
            print(f"❌ Error processing file {file}: {e}. Skipping.")

    if not docs:
         print("🛑 --- Document Loading Finished: No processable documents were loaded. ---")
         return []

    print("-" * 50) # <-- ADDED FOR VISIBILITY
    print("Splitting documents into smaller chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=75)
    split_docs = text_splitter.split_documents(docs)
    
    # --- NEW: Final summary print ---
    print("\n" + "="*50)
    print("✅ --- Document Loading Summary --- ✅")
    print(f"Total documents successfully loaded: {len(docs)}")
    print(f"Total chunks created for the vector store: {len(split_docs)}")
    print("="*50 + "\n")
    # ----------------------------------
    
    return split_docs