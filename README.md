# 🎓 Admission RAG Assistant

An AI-powered Retrieval-Augmented Generation (RAG) application that answers questions about Army Institute of Technology admission using official prospectus documents.

## Features

- 📄 Document-based Q&A from official admission documents
- 🤖 Powered by Groq's LLaMA 3.1-8B model
- 🔍 Semantic search with HuggingFace embeddings
- 💬 Interactive Streamlit chat interface
- 🚀 Production-ready deployment

## Prerequisites

- Python 3.8+
- Groq API key (get it from [console.groq.com](https://console.groq.com))
- PROSPECTUS-2025.pdf file in the project root

## Installation

1. **Clone the repository**
   ```bash
   git clone <repo-url>
   cd QA_Assistant
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env and add your GROQ_API_KEY
   ```

## Running Locally

```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`

## Deployment Options

### Streamlit Cloud (Recommended)
1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Deploy by connecting your GitHub repo
4. Add `GROQ_API_KEY` in Streamlit Cloud secrets

### Docker Deployment
```bash
docker build -t admission-assistant .
docker run -p 8501:8501 -e GROQ_API_KEY=your_key admission-assistant
```

### Heroku Deployment
```bash
heroku login
heroku create your-app-name
git push heroku main
```

## Configuration

Edit parameters in `app.py`:
- `SIMILARITY_THRESHOLD`: Adjust relevance filtering (default: 0.25)
- `temperature`: LLM creativity (0.0-1.0, default: 0.2)
- `max_tokens`: Response length (default: 350)
- `similarity_top_k`: Number of retrieved documents (default: 8)

## Troubleshooting

- **"GROQ_API_KEY not found"**: Ensure `.env` file is properly set up
- **"Document not found"**: Place `PROSPECTUS-2025.pdf` in project root
- **Memory issues**: The app requires ~2GB RAM minimum

## License

MIT