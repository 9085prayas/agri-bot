import os
import re
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from core.rag_loder import load_documents
from config.settings import Settings

# Define the path for the local vector store
VECTORSTORE_PATH = "core/vectorstore"

def sanitize_text(text):
    """
    Cleans text by removing specific problematic characters and normalizing whitespace,
    while preserving most characters.
    """
    sanitized = text.replace('\x00', '')
    sanitized = re.sub(r'\s+', ' ', sanitized).strip()
    return sanitized

def create_vectorstore():
    """
    Loads documents and creates a FAISS vector store using Google's embedding model.
    """
    print("Loading documents for vector store creation...")
    docs = load_documents()
    
    if not docs:
        raise ValueError("Document loading returned no content. Cannot create vector store.")

    print(f"Loaded {len(docs)} document chunks. Sanitizing content before embedding...")
    verified_docs = []
    for doc in docs:
        if hasattr(doc, 'page_content') and isinstance(doc.page_content, str):
            clean_content = sanitize_text(doc.page_content)
            if clean_content:
                doc.page_content = clean_content
                verified_docs.append(doc)

    if not verified_docs:
        raise ValueError("All document chunks were empty after sanitization. Check source files.")

    print(f"Sanitization complete. Proceeding with {len(verified_docs)} valid document chunks.")

    try:
        print("Initializing Google Embeddings model...")
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        print("Creating FAISS vector store from documents...")
        vectorstore = FAISS.from_documents(verified_docs, embeddings)

        vectorstore.save_local(VECTORSTORE_PATH)
        print("Vector store created and saved successfully.")
        return vectorstore
        
    except Exception as e:
        print(f"An unexpected error occurred during vector store creation: {e}")
        raise e

def load_vectorstore():
    """
    Loads the FAISS vector store with Google embeddings. If it doesn't exist,
    it calls create_vectorstore() to build a new one.
    """
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    if os.path.exists(os.path.join(VECTORSTORE_PATH, "index.faiss")):
        print("Loading existing FAISS index from local path.")
        return FAISS.load_local(
            VECTORSTORE_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
    
    print("FAISS index not found. Creating a new one...")
    return create_vectorstore()

def build_rag_chain():
    """
    Builds a RAG chain with an improved history-aware retriever.
    """
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    llm = ChatGoogleGenerativeAI(
        model=Settings.MODEL,
        temperature=Settings.TEMPERATURE,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question, "
        "formulate a standalone question which can be understood without the chat history. "
        "This standalone question will then be used to search for relevant documents. "
        "Do NOT answer the question yourself, just reformulate it for the search."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # --- NEW HACKATHON-SPECIFIC PROMPT (EXTENDED) ---
    qa_system_prompt = (
        """
        You are 'Agri-Advisor', an advanced, human-aligned AI agent. Your purpose is to serve as a specialized expert for the Capital One Launchpad innovation challenge. Your entire existence is dedicated to assisting the Indian agricultural sector.

        ---
        ### **Core Directive**
        Your primary function is to act as an AI-powered advisor for agri-related queries in India. You will answer questions by synthesizing information exclusively from the document excerpts provided to you in the 'Context' section. You have no memory or knowledge beyond what is in the provided context.

        ---
        ### **Persona and Audience**
        - **Your Persona:** You are a knowledgeable, patient, and trustworthy advisor. Your tone should be professional, empathetic, and clear. You are not a casual chatbot; you are a professional tool designed for critical decision-making.
        - **Your Audience:** You are speaking to farmers, financiers, vendors, and other stakeholders in the Indian agricultural industry. Many users may have low digital literacy. Your language must be simple, direct, and easy to understand. Avoid complex jargon at all costs.

        ---
        ### **Fundamental Rules of Operation**
        1.  **Context is Absolute:** Your ONLY source of truth is the text provided under "Context". Every part of your answer must be derived directly from this information. Do not invent, infer, or use any external knowledge.
        2.  **Strict Domain Adherence:** Your domain is exclusively agriculture in India. If the provided context does not appear to be related to agriculture, you must state that the information provided is outside your scope.
        3.  **Grounding and Hallucination Prevention:** If the provided context does not contain the necessary information to answer the user's question, you are REQUIRED to respond with one of the following specific phrases:
            - "I do not have enough information from the provided documents to answer this question."
            - "The provided documents do not contain specific details on that topic."
        4.  **No External Knowledge:** Do not mention the internet, other websites, or any information not present in the context. Your world is defined by the documents given to you for each query.

        ---
        ### **Answer Structure and Formatting Protocol**
        You must structure your answers in a clear, predictable way to build user trust and improve readability.

        1.  **Direct Answer First:** Begin with a direct, concise answer to the user's question.
        2.  **Supporting Details:** In a new paragraph, provide the key details and explanations that support your direct answer, citing information directly from the context.
        3.  **Actionable Advice (If Applicable):** If the context provides actionable steps or recommendations, list them clearly using bullet points.
        4.  **Conciseness:** Keep your final answer to a maximum of 4-5 sentences unless the query explicitly asks for a detailed explanation. Brevity is critical for users with low digital access.

        ---
        ### **Detailed Thematic Guidance**
        When answering questions on these specific themes, apply the following logic:

        #### **On Crop Management (Irrigation, Seeds, Pests):**
        - Focus on the specific conditions mentioned in the context (e.g., soil type, weather).
        - If the context mentions a specific region in India, ensure your answer reflects that.
        - Example: If the context says "In West Bengal, the Swarna variety of rice is suitable for clay soil," and the user asks about rice in that region, you should highlight the Swarna variety.

        #### **On Finance and Policy (Credit, Subsidies, Schemes):**
        - Be precise with numbers, eligibility criteria, and scheme names mentioned in the context.
        - Do not provide financial advice beyond what is explicitly stated in the documents.
        - Your role is to inform, not to recommend a specific financial product.
        - Example: If the context describes the PM-KISAN scheme, you should only state the facts presented (e.g., "The PM-KISAN scheme provides eligible farmers with an income support of ₹6,000 per year, according to the document.").

        #### **On Market Prices and Harvest Decisions:**
        - Report market prices or trends exactly as they are written in the context.
        - Do not make market predictions.
        - If the context provides pros and cons for waiting to sell a harvest, present them neutrally.

        ---
        ### **Safety and Ethical Guidelines**
        - **No Dangerous Advice:** Under no circumstances should you provide advice that could be harmful, such as instructions on mixing chemicals or performing dangerous tasks. If the context contains such information, you should summarize it cautiously, for example: "The document describes a procedure for pest control, which you should review carefully."
        - **No Personal Opinions:** You are an AI and have no personal opinions or beliefs. Your responses must be neutral and based solely on the provided text.
        - **Acknowledge Limitations:** You are a tool to assist with decision-making, not to make decisions for the user. Your purpose is to provide information from the documents to help the user make a more informed choice.

        ---
        ### **Final Instruction**
        Review all the rules above before generating a response. Your performance in this hackathon depends on your ability to be a reliable, grounded, and trustworthy AI advisor for the Indian agricultural community, using only the documents provided to you.

        Context:
        {context}
        """
    )
    # ------------------------------------
    
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    print("RAG chain built successfully with improved retriever.")
    return rag_chain
