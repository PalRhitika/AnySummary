import os
from dotenv import load_dotenv
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, RetrievalQA
import streamlit as st
load_dotenv()
api_key=os.getenv('OPENAI_API_KEY')


class Document:
    def __init__(self, page_content, metadata,id):
        self.page_content = page_content
        self.metadata = metadata
        self.id=id


def read_pdf(file):
    documents=[]

    reader = PyPDF2.PdfReader(file)
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text.strip():  # Avoid empty pages
            document = Document(
                    page_content=text,
                    metadata={"page_number": i + 1, "source": file},
                    id={i+1}
                )
            documents.append(document)
    return documents

def split_text(documents):
    doc_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, separators=('\n\n', '\n', " "))
    splitted_doc=doc_splitter.split_documents(documents)
    return splitted_doc


def create_vector_database(splitted_doc):
    embeddings = OpenAIEmbeddings(api_key=api_key)
    faiss_db = FAISS.from_documents(splitted_doc, embeddings)
    return faiss_db

 #for summarization
def retrieve_relevant_chunks(faiss_db, num_chunks=5):
    retriever = faiss_db.as_retriever(search_type="similarity", search_kwargs={"k": num_chunks})
    query = "Summarize the key points of this document."
    retrieved_chunks = retriever.get_relevant_documents(query)
    return retrieved_chunks

def summarize_chunks(retrieved_chunks):
    llm=ChatOpenAI(temperature=0.2,api_key=api_key, model="gpt-3.5-turbo")
    prompt_template = """
        You are a helpful assistant that summarizes text in gives 5 concise paragraphs. Refine the following summary with this new text:
        Summary so far:
        {summary}

        New text:
        {new_text}

        Updated Summary:
        """

    # Create the prompt template
    prompt = PromptTemplate(
            input_variables=["summary", "new_text"],
            template=prompt_template
        )

    # Create the LLMChain
    chain = LLMChain(llm=llm, prompt=prompt)
    summary = ""
    for document in retrieved_chunks:
            # Call the chain for each chunk to refine the summary
            new_text = document.page_content
            summary = chain.run({"summary": summary, "new_text": new_text})
    return summary


#for Q/A chatbot

def build_retrieval_qa(faiss_db):
    retriever = faiss_db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    llm = ChatOpenAI(temperature=0.2, api_key=api_key, model="gpt-3.5-turbo")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True  # Optional, if you want source docs for the response
    )
    return qa_chain


# Streamlit App
st.set_page_config(layout="wide", page_title="PDF Summarizer with searchable query")

st.sidebar.title("PDF Upload")
uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])

# Layout placeholders
summary_placeholder = st.empty()
qa_placeholder = st.empty()

if uploaded_file:
    st.sidebar.success("File uploaded successfully!")
    with st.spinner("Processing the file..."):
        # Process the PDF
        documents = read_pdf(uploaded_file)
        splitted_doc = split_text(documents)
        faiss_db = create_vector_database(splitted_doc)
        retrieved_chunks = retrieve_relevant_chunks(faiss_db, num_chunks=10)
        summary_ready = True

    st.sidebar.success("Processing complete. You can now summarize the code.Please click on summarize bottom")
else:
    summary_ready = False

col1, col2 = st.columns(2)
# Middle Section: Summarization
with col1:
    st.title("Summarization")
    st.write("Click the button below to summarize the uploaded document.")
    summarize_button = st.button("Summarize", disabled=not summary_ready)
    if summarize_button:
        with st.spinner("Generating summary..."):
            summary = summarize_chunks(retrieved_chunks)
            st.session_state.generated_summary = summary
    if 'generated_summary' in st.session_state:
        st.subheader("Generated Summary")
        st.write(st.session_state.generated_summary)

# Right Section: Q&A Chatbot
with col2:
    st.title("Ask the doc app:")
    query_input = st.text_input("Ask a question about the document:", disabled=not summary_ready)
    qa_button = st.button("Ask", disabled=not summary_ready)
    if qa_button and query_input:
        if "summarize" in query_input.lower():
            st.warning("For summary, please click on the 'Summarize' button.")
        else:
            with st.spinner("Fetching answer..."):
                qa_chain = build_retrieval_qa(faiss_db)
                response = qa_chain(query_input)
            st.subheader("Answer")
            st.write(response["result"])
            st.subheader("Sources")
            for doc in response["source_documents"]:
                st.write(f"- Page {doc.metadata['page_number']}: {doc.page_content[:200]}...")

