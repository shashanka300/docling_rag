import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any

from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from sentence_transformers import SentenceTransformer
from langchain_ibm import WatsonxLLM
import lancedb
from tempfile import mkdtemp

class DocumentProcessor:
    def __init__(self, api_key: str, project_id: str):
        """Initialize document processor with necessary components"""
        self.api_key = api_key
        self.project_id = project_id
        self.setup_document_converter()
        self.setup_ml_components()
        
    def setup_document_converter(self):
        """Configure document converter with advanced processing capabilities"""
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.do_cell_matching = True
        pipeline_options.ocr_options.lang = ["en"]
        pipeline_options.accelerator_options = AcceleratorOptions(
            num_threads=4,
            device=AcceleratorDevice.MPS
        )

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                    backend=PyPdfiumDocumentBackend
                )
            }
        )
        
    def setup_ml_components(self):
        """Initialize embedding model and LLM"""
        self.embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.llm = WatsonxLLM(
            model_id="mistralai/mixtral-8x7b-instruct-v01",
            url='https://us-south.ml.cloud.ibm.com',
            apikey=self.api_key,
            project_id=self.project_id,
            params={"max_new_tokens": 2000}
        )

    def extract_chunk_metadata(self, chunk) -> Dict[str, Any]:
        """Extract essential metadata from a chunk"""
        metadata = {
            "text": chunk.text,
            "headings": [],
            "page_info": None,
            "content_type": None
        }
        
        if hasattr(chunk, 'meta'):
            # Extract headings
            if hasattr(chunk.meta, 'headings') and chunk.meta.headings:
                metadata["headings"] = chunk.meta.headings
            
            # Extract page information and content type
            if hasattr(chunk.meta, 'doc_items'):
                for item in chunk.meta.doc_items:
                    if hasattr(item, 'label'):
                        metadata["content_type"] = str(item.label)
                    
                    if hasattr(item, 'prov') and item.prov:
                        for prov in item.prov:
                            if hasattr(prov, 'page_no'):
                                metadata["page_info"] = prov.page_no
        
        return metadata

    def process_document(self, pdf_path: str) -> Any:
        """Process document and create searchable index with metadata"""
        print(f"Processing document: {pdf_path}")
        start_time = time.time()
        
        # Convert document
        result = self.converter.convert(pdf_path)
        doc = result.document
        
        # Create chunks using hybrid chunker
        chunker = HybridChunker(tokenizer="jinaai/jina-embeddings-v3")
        chunks = list(chunker.chunk(doc))
        
        # Process chunks and extract metadata
        print("\nProcessing chunks:")
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            metadata = self.extract_chunk_metadata(chunk)
            processed_chunks.append(metadata)
            
            # Print chunk information for inspection
            # print(f"\nChunk {i}:")
            # if metadata['headings']:
            #     print(f"Section: {' > '.join(metadata['headings'])}")
            # print(f"Page: {metadata['page_info']}")
            # print(f"Type: {metadata['content_type']}")
            # print("-" * 40)
        
        # Create vector database
        print("\nCreating vector database...")
        db_uri = str(Path(mkdtemp()) / "docling.db")
        self.db = lancedb.connect(db_uri)
        
        # Store chunks with embeddings and metadata
        data = []
        for chunk in processed_chunks:
            embeddings = self.embed_model.encode(chunk['text'])
            data_item = {
                "vector": embeddings,
                "text": chunk['text'],
                "headings": json.dumps(chunk['headings']),
                "page": chunk['page_info'],
                "content_type": chunk['content_type']
            }
            data.append(data_item)
        
        self.index = self.db.create_table("document_chunks", data=data, exist_ok=True)
        
        processing_time = time.time() - start_time
        print(f"\nDocument processing completed in {processing_time:.2f} seconds")
        return self.index

    def format_context(self, chunks: List[Dict]) -> str:
        """Format retrieved chunks into a structured context for the LLM"""
        context_parts = []
        for chunk in chunks:
            # Include section information if available
            try:
                headings = json.loads(chunk['headings'])
                if headings:
                    context_parts.append(f"Section: {' > '.join(headings)}")
            except:
                pass

            # Add page reference
            if chunk['page']:
                context_parts.append(f"Page {chunk['page']}:")

            # Add the content
            context_parts.append(chunk['text'])
            context_parts.append("-" * 40)

        return "\n".join(context_parts)

    def query(self, question: str, k: int = 5) -> str:
        """Query the document using semantic search and generate an answer"""
        # Create query embedding and search
        query_embedding = self.embed_model.encode(question)
        results = self.index.search(query_embedding).limit(k)
        chunks = results.to_pandas()
        
        # Display retrieved chunks with their context
        print(f"\nRelevant chunks for query: '{question}'")
        print("=" * 80)
        
        # Format chunks for display and LLM
        context = self.format_context(chunks.to_dict('records'))
        print(context)
        
        # Generate answer using structured context
        prompt = f"""Based on the following excerpts from a document:

{context}

Please answer this question: {question}

Make use of the section information and page numbers in your answer when relevant.
"""
        
        return self.llm(prompt)

def main():
    logging.basicConfig(level=logging.INFO)
    
    processor = DocumentProcessor(
        api_key="your-api-key",
        project_id="your-project-id"
    )
    
    # Process document
    pdf_path = "your_document.pdf"
    processor.process_document(pdf_path)
    
    # Example query
    question = "What are the main features of InternLM-XComposer-2.5?"
    answer = processor.query(question)
    print("\nAnswer:")
    print("=" * 80)
    print(answer)

if __name__ == "__main__":
    main()
