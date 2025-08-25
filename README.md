# Research Paper Summarizer

A comprehensive AI-powered system for ingesting, processing, and querying research papers with advanced RAG (Retrieval-Augmented Generation) capabilities.

## Features

- **Document Ingestion**: Upload and process PDF research papers
- **Intelligent Chunking**: Advanced text segmentation for optimal retrieval
- **Vector Embeddings**: High-quality document embeddings for semantic search
- **RAG Pipeline**: Retrieve relevant context and generate accurate summaries
- **Query Interface**: Ask questions about uploaded papers
- **Safety & Evaluation**: Content safety checks and response quality evaluation
- **Observability**: Comprehensive logging and monitoring
- **Docker Support**: Containerized deployment with PostgreSQL and pgvector

## Tech Stack

- **Backend**: FastAPI (Python)
- **Database**: PostgreSQL with pgvector extension
- **Embeddings**: OpenAI embeddings
- **LLM**: OpenAI GPT models
- **Containerization**: Docker & Docker Compose
- **Testing**: pytest

## Project Structure

```
├── app/
│   ├── api/                 # API endpoints
│   │   ├── datasets.py      # Dataset management
│   │   ├── eval.py          # Evaluation endpoints
│   │   ├── health.py        # Health checks
│   │   ├── ingest.py        # Document ingestion
│   │   ├── query.py         # Query processing
│   │   └── review.py        # Content review
│   ├── services/            # Business logic
│   │   ├── chunking.py      # Text chunking strategies
│   │   ├── datasets.py      # Dataset operations
│   │   ├── embeddings.py    # Embedding generation
│   │   ├── llm.py           # LLM interactions
│   │   ├── observability.py # Logging & monitoring
│   │   ├── parsing.py       # Document parsing
│   │   ├── retrieval.py     # RAG retrieval
│   │   └── safety.py        # Content safety
│   ├── database.py          # Database configuration
│   ├── main.py             # FastAPI application
│   ├── models.py           # Database models
│   └── settings.py         # Configuration
├── tests/                  # Test suite
├── docker-compose.yml      # Docker services
├── Dockerfile             # Application container
└── requirements.txt       # Python dependencies
```

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Python 3.11+ (for local development)
- OpenAI API key

### Environment Setup

1. Clone the repository:
```bash
git clone https://github.com/pnehete23/research-paper-summarizer.git
cd research-paper-summarizer
```

2. Create environment file:
```bash
cp .env.example .env
```

3. Configure your `.env` file:
```env
OPENAI_API_KEY=your_openai_api_key_here
DATABASE_URL=postgresql://postgres:password@localhost:5432/research_papers
```

### Docker Deployment

1. Start the services:
```bash
docker-compose up -d
```

2. The API will be available at `http://localhost:8000`

3. Access API documentation at `http://localhost:8000/docs`

### Local Development

1. Create virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Start PostgreSQL (via Docker):
```bash
docker-compose up -d postgres
```

4. Run the application:
```bash
uvicorn app.main:app --reload
```

## API Endpoints

### Core Functionality
- `POST /api/ingest` - Upload and process research papers
- `POST /api/query` - Query processed documents
- `GET /api/datasets` - List available datasets
- `DELETE /api/datasets/{dataset_id}` - Delete dataset

### Evaluation & Safety
- `POST /api/eval/evaluate` - Evaluate system performance
- `POST /api/review/safety` - Content safety review
- `GET /api/health` - Health check

## Usage Examples

### Ingest a Research Paper
```python
import requests

files = {'file': open('research_paper.pdf', 'rb')}
data = {'dataset_name': 'ml_papers'}

response = requests.post(
    'http://localhost:8000/api/ingest',
    files=files,
    data=data
)
```

### Query Documents
```python
import requests

query_data = {
    'query': 'What are the main findings of this paper?',
    'dataset_name': 'ml_papers'
}

response = requests.post(
    'http://localhost:8000/api/query',
    json=query_data
)
```

## Testing

Run the test suite:
```bash
pytest tests/
```

Run with coverage:
```bash
pytest tests/ --cov=app
```

## Database Schema

The system uses PostgreSQL with pgvector extension for storing:
- **Documents**: Metadata and content of uploaded papers
- **Chunks**: Segmented text portions with embeddings
- **Datasets**: Logical grouping of related documents
- **Queries**: Search history and results

## Configuration

Key configuration options in `app/settings.py`:
- **Chunking Strategy**: Overlap size, chunk size
- **Embedding Model**: OpenAI model selection
- **LLM Settings**: Model, temperature, max tokens
- **Database**: Connection settings
- **Safety**: Content filtering thresholds

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Troubleshooting

### Common Issues

1. **Database Connection Error**
   - Ensure PostgreSQL is running
   - Verify database credentials in `.env`

2. **OpenAI API Error**
   - Check your API key is valid
   - Verify you have sufficient credits

3. **PDF Processing Error**
   - Ensure the PDF is readable
   - Check file size limits

### Logs

Application logs are available in the container:
```bash
docker-compose logs app
```

## Roadmap

- [ ] Support for additional document formats (Word, HTML)
- [ ] Multi-language support
- [ ] Advanced query operators
- [ ] Web interface
- [ ] Batch processing capabilities
- [ ] Custom embedding models
- [ ] API rate limiting
- [ ] User authentication