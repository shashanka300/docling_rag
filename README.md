# Advanced Document Processing and RAG System Using Docling

This repository implements a sophisticated Retrieval-Augmented Generation (RAG) system using Docling for advanced document processing. The system combines state-of-the-art document analysis with semantic search and language model capabilities to create an intelligent document question-answering system.

## Features

The system provides comprehensive document processing and question-answering capabilities through several key features:

Document Processing: The implementation leverages Docling's advanced processing capabilities, including OCR for text extraction from images, table structure analysis, and sophisticated document layout understanding. This ensures that all document content, including text in images and complex tables, is properly processed.

Intelligent Chunking: The system uses Docling's hybrid chunking strategy to break documents into meaningful segments while preserving context and document structure. This approach maintains the semantic coherence of document sections while optimizing for retrieval.

Semantic Search: Through the integration of modern embedding models and vector search capabilities, the system can find the most relevant document sections for any given query. This ensures accurate information retrieval that goes beyond simple keyword matching.

Context-Aware Responses: The system preserves document structure, including section headings and page numbers, allowing for more contextual and well-referenced answers to queries.

Flexible LLM Support: The system supports both WatsonX and Ollama, allowing users to choose their preferred language model backend based on their needs and resources.

## Installation

First, ensure you have Python 3.8 or higher installed. Then install the required dependencies:

```bash
# Install base requirements
pip install docling sentence-transformers lancedb

# For WatsonX support
pip install langchain-ibm

# For Ollama support
pip install langchain ollama
```

If you plan to use Ollama, you'll need to install it separately:

```bash
# On macOS
brew install ollama

# On Linux
curl -fsSL https://ollama.com/install.sh | sh

# After installation, pull your preferred model
ollama pull mistral  # or another model of your choice
```

## Configuration

The system supports two LLM backends. Choose the one that best fits your needs:

### Option 1: WatsonX Configuration

```python
from langchain_ibm import WatsonxLLM

llm = WatsonxLLM(
    model_id="mistralai/mixtral-8x7b-instruct-v01",
    url='https://us-south.ml.cloud.ibm.com',
    apikey="your-watsonx-api-key",
    project_id="your-project-id",
    params={"max_new_tokens": 2000}
)
```

### Option 2: Ollama Configuration

```python
from langchain.llms import Ollama

llm = Ollama(
    model="mistral",  # or any other model you've pulled
    temperature=0.1,
    num_ctx=2048
)
```

## Usage

Here's how to use the system with either LLM backend:

```python
from document_processor import DocumentProcessor

# Initialize with WatsonX
processor = DocumentProcessor(
    llm_type="watsonx",
    api_key="your-api-key",
    project_id="your-project-id"
)

# Or initialize with Ollama
processor = DocumentProcessor(
    llm_type="ollama",
    model_name="mistral"  # or your preferred model
)

# Process a document
processor.process_document("path/to/your/document.pdf")

# Ask questions
answer = processor.query("What are the main points of the document?")
print(answer)
```

## Implementation Details

The DocumentProcessor class now supports both LLM backends through a flexible initialization:

```python
def setup_ml_components(self, llm_type: str = "ollama", **kwargs):
    """Initialize embedding model and LLM"""
    self.embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    if llm_type == "watsonx":
        self.llm = WatsonxLLM(
            model_id="mistralai/mixtral-8x7b-instruct-v01",
            url='https://us-south.ml.cloud.ibm.com',
            apikey=kwargs.get('api_key'),
            project_id=kwargs.get('project_id'),
            params={"max_new_tokens": 2000}
        )
    elif llm_type == "ollama":
        self.llm = Ollama(
            model=kwargs.get('model_name', 'mistral'),
            temperature=kwargs.get('temperature', 0.1),
            num_ctx=kwargs.get('num_ctx', 2048)
        )
    else:
        raise ValueError(f"Unsupported LLM type: {llm_type}")
```


## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This implementation builds upon several powerful tools and libraries:
- Docling for advanced document processing
- SentenceTransformers for embeddings
- LanceDB for vector search
- WatsonX and Ollama for language model capabilities

For more information about the underlying technologies, consult their respective documentation.
