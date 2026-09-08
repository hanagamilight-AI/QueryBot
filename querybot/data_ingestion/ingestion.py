"""
Data ingestion pipeline for political intelligence data
Handles elections, surveys, social media, and manifesto documents
"""
import json
import csv
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from loguru import logger
import pandas as pd

from config.settings import settings
from database.models import Document, DocumentChunk, SessionLocal, init_db
from embeddings.embedder import get_embedding_service
from data_ingestion.chunking import chunk_document


class DataIngestionPipeline:
    """
    Pipeline for ingesting and processing political intelligence data
    """
    
    def __init__(self):
        self.embedding_service = get_embedding_service()
        self.session = SessionLocal()
    
    def ingest_from_directory(self, directory: str, source_type: str) -> int:
        """
        Ingest all supported files from a directory
        
        Args:
            directory: Path to data directory
            source_type: Type of data (election, survey, social_media, manifesto)
            
        Returns:
            Number of documents ingested
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            logger.warning(f"Directory does not exist: {directory}")
            return 0
        
        total_ingested = 0
        
        for file_path in dir_path.iterdir():
            if file_path.is_file():
                ext = file_path.suffix.lower()
                if ext in settings.SUPPORTED_FORMATS:
                    try:
                        count = self.ingest_file(file_path, source_type)
                        total_ingested += count
                        logger.info(f"Ingested {count} documents from {file_path}")
                    except Exception as e:
                        logger.error(f"Error ingesting {file_path}: {e}")
        
        return total_ingested
    
    def ingest_file(self, file_path: Path, source_type: str) -> int:
        """
        Ingest a single file based on its format
        
        Args:
            file_path: Path to the file
            source_type: Type of data source
            
        Returns:
            Number of documents ingested
        """
        ext = file_path.suffix.lower()
        
        if ext == ".json":
            return self._ingest_json(file_path, source_type)
        elif ext == ".csv":
            return self._ingest_csv(file_path, source_type)
        elif ext == ".txt":
            return self._ingest_text(file_path, source_type)
        elif ext == ".pdf":
            return self._ingest_pdf(file_path, source_type)
        else:
            logger.warning(f"Unsupported file format: {ext}")
            return 0
    
    def _ingest_json(self, file_path: Path, source_type: str) -> int:
        """Ingest JSON file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle both single object and list of objects
        if isinstance(data, dict):
            data = [data]
        
        count = 0
        for item in data:
            doc = self._create_document(item, source_type, str(file_path))
            if doc:
                self._save_document(doc)
                count += 1
        
        return count
    
    def _ingest_csv(self, file_path: Path, source_type: str) -> int:
        """Ingest CSV file"""
        df = pd.read_csv(file_path)
        count = 0
        
        for _, row in df.iterrows():
            doc_dict = row.to_dict()
            doc = self._create_document(doc_dict, source_type, str(file_path))
            if doc:
                self._save_document(doc)
                count += 1
        
        return count
    
    def _ingest_text(self, file_path: Path, source_type: str) -> int:
        """Ingest plain text file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        metadata = {
            "filename": file_path.name,
            "filepath": str(file_path)
        }
        
        doc = Document(
            content=content,
            source_type=source_type,
            source_id=str(file_path),
            language="en",
            document_date=datetime.fromtimestamp(file_path.stat().st_mtime),
            metadata_json=metadata
        )
        
        self._save_document(doc)
        return 1
    
    def _ingest_pdf(self, file_path: Path, source_type: str) -> int:
        """Ingest PDF file (requires additional library)"""
        try:
            import PyPDF2
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
            
            metadata = {
                "filename": file_path.name,
                "filepath": str(file_path),
                "pages": len(reader.pages)
            }
            
            doc = Document(
                content=text,
                source_type=source_type,
                source_id=str(file_path),
                language="en",
                document_date=datetime.fromtimestamp(file_path.stat().st_mtime),
                metadata_json=metadata
            )
            
            self._save_document(doc)
            return 1
        except ImportError:
            logger.warning("PyPDF2 not installed. Install with: pip install PyPDF2")
            return 0
    
    def _create_document(self, data: Dict[str, Any], source_type: str, source_id: str) -> Optional[Document]:
        """
        Create a Document object from raw data
        
        Args:
            data: Raw data dictionary
            source_type: Type of data source
            source_id: Identifier for the source
            
        Returns:
            Document object or None if invalid
        """
        # Extract content - look for common content fields
        content_fields = ['content', 'text', 'body', 'message', 'description']
        content = None
        for field in content_fields:
            if field in data and data[field]:
                content = str(data[field])
                break
        
        if not content:
            logger.warning(f"No content found in document from {source_id}")
            return None
        
        # Generate embedding
        embedding = self.embedding_service.embed_text(content)
        
        # Extract metadata fields
        doc = Document(
            content=content,
            embedding=embedding.tolist(),
            source_type=source_type,
            source_id=source_id,
            constituency=data.get('constituency') or data.get('constituency_name'),
            state=data.get('state') or data.get('state_name'),
            party=data.get('party') or data.get('party_name'),
            candidate=data.get('candidate') or data.get('candidate_name'),
            language=data.get('language', 'en'),
            sentiment_score=data.get('sentiment_score') or data.get('sentiment'),
            document_date=self._parse_date(data.get('date') or data.get('document_date')),
            metadata_json={k: v for k, v in data.items() 
                          if k not in ['content', 'text', 'body', 'message']}
        )
        
        return doc
    
    def _save_document(self, doc: Document):
        """Save document and its chunks to database"""
        try:
            # Save main document
            self.session.add(doc)
            self.session.commit()
            
            # Create and save chunks
            chunks = chunk_document(doc.content)
            for i, chunk_text in enumerate(chunks):
                chunk_embedding = self.embedding_service.embed_text(chunk_text)
                chunk = DocumentChunk(
                    document_id=doc.id,
                    chunk_index=i,
                    content=chunk_text,
                    embedding=chunk_embedding.tolist()
                )
                self.session.add(chunk)
            
            self.session.commit()
            logger.debug(f"Saved document {doc.id} with {len(chunks)} chunks")
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error saving document: {e}")
            raise
    
    def _parse_date(self, date_value: Any) -> Optional[datetime]:
        """Parse date from various formats"""
        if date_value is None:
            return None
        
        if isinstance(date_value, datetime):
            return date_value
        
        if isinstance(date_value, str):
            # Try multiple formats
            formats = [
                "%Y-%m-%d",
                "%d-%m-%Y",
                "%Y/%m/%d",
                "%d/%m/%Y",
                "%B %d, %Y",
                "%b %d, %Y"
            ]
            for fmt in formats:
                try:
                    return datetime.strptime(date_value, fmt)
                except ValueError:
                    continue
        
        return None
    
    def close(self):
        """Close the session"""
        self.session.close()


def run_ingestion_pipeline():
    """Run the complete ingestion pipeline"""
    logger.info("Starting data ingestion pipeline...")
    
    # Initialize database
    init_db()
    
    # Create pipeline
    pipeline = DataIngestionPipeline()
    
    try:
        # Ingest from all configured directories
        total = 0
        for source_dir in settings.DATA_SOURCE_DIRS:
            source_type = source_dir.split('/')[-1]  # Extract type from path
            count = pipeline.ingest_from_directory(source_dir, source_type)
            total += count
            logger.info(f"Ingested {count} documents from {source_dir}")
        
        logger.info(f"Ingestion complete. Total documents: {total}")
        return total
    finally:
        pipeline.close()


if __name__ == "__main__":
    run_ingestion_pipeline()
