import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document

# Load API key from Streamlit secrets
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

st.title("Document Analysis Chatbot")

# Main area for file upload
st.header("Upload Your Document")
file = st.file_uploader("Upload a PDF (e.g., license agreement, research paper)", type="pdf")

# Process the uploaded PDF and generate summary
if file is not None:
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""  # Handle potential None values

    # Convert text to a Document object for summarization
    doc = Document(page_content=text)

    # Split text into chunks for Q&A
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n", r"\."],  # Use raw string for the dot to avoid SyntaxWarning
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    # Generate embeddings and create vector store
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vector_store = FAISS.from_texts(chunks, embeddings)

    # Initialize the LLM
    llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model_name="gpt-3.5-turbo",  # Using gpt-3.5-turbo for broader access
        temperature=0.2,
        max_tokens=1500
    )

    # Generate summary
    summarize_chain = load_summarize_chain(llm, chain_type="stuff")
    summary = summarize_chain.run([doc])  # Pass a list of Document objects
    st.subheader("Document Summary")
    st.write(summary)

    # Search bar for questions
    st.subheader("Search the Document")
    search_query = st.text_input("Enter your search query here:", placeholder="e.g., What are the main terms?")
    
    if search_query:
        # Perform similarity search
        docs = vector_store.similarity_search(search_query, k=3)  # Top 3 relevant chunks

        # Load Q&A chain
        qa_chain = load_qa_chain(llm, chain_type="stuff")
        response = qa_chain.run(input_documents=docs, question=search_query)

        st.subheader("Search Results")
        st.write(response)

# Add instructions
st.sidebar.info("Upload a PDF to start analyzing. Use the search bar to ask questions about the content!")

# Custom CSS for better UI
st.markdown("""
    <style>
    .stFileUploader > div {
        padding: 10px;
        border: 2px dashed #4CAF50;
        border-radius: 5px;
        text-align: center;
    }
    .stTextInput > div > input {
        width: 100%;
        padding: 10px;
        font-size: 16px;
        border-radius: 5px;
        border: 1px solid #ccc;
    }
    </style>
""", unsafe_allow_html=True)