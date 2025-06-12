# 📄 AnySummary

**AnySummary** is a powerful PDF summarization and question-answering app built with **Python**, **Streamlit**, and **OpenAI's GPT model**. It enables users to upload a **PDF document**, generate a summary, and even **ask questions** about the content — perfect for quickly understanding large files like research papers, reports, or manuals.

## 🚀 Features

- 📄 **PDF Summarization**: Upload any PDF document and receive a clear, structured summary.
- ❓ **Ask the Doc**: Ask questions about your uploaded PDF.
- ⚙️ Powered by **OpenAI's GPT** for high-quality, human-like responses.
- 🖥️ Streamlit frontend for a clean and interactive user experience.

## 🧪 Try It Out

A sample PDF is included in the repository — feel free to use it to test the summarizer and Q&A features!

## 💻 Tech Stack

- Python 3.x
- Streamlit
- OpenAI GPT (via API)
- `PyPDF2` for PDF parsing
- Vector search using `FAISS` or similar for Q&A

## 📦 Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/PalRhitika/AnySummary.git
   cd AnySummary
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set your OpenAI API key:
   ```bash
   export OPENAI_API_KEY=your_key_here  # or use a .env file
   ```

5. Run the app:
   ```bash
   streamlit run app.py
   ```


