import sqlite3
import chromadb
from datetime import datetime
from typing import List, Dict, Optional

class ContentDatabase:
    def __init__(self, sqlite_path: str, chroma_path: str):
        self.sqlite_path = sqlite_path
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.chroma_client.get_or_create_collection("content_inventory")
        self._init_sqlite()
    
    def _init_sqlite(self):
        """Initialize SQLite database with content metadata table"""
        conn = sqlite3.connect(self.sqlite_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS content_items (
                id TEXT PRIMARY KEY,
                url TEXT,
                title TEXT,
                summary TEXT,
                content_type TEXT,
                word_count INTEGER,
                created_at TIMESTAMP,
                updated_at TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()
    
    def store_content(self, content_id: str, metadata: Dict, chunks: List[str], embeddings: List[List[float]]):
        """Store content in both SQLite and Chroma"""
        # Store metadata in SQLite
        conn = sqlite3.connect(self.sqlite_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO content_items 
            (id, url, title, summary, content_type, word_count, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            content_id,
            metadata.get('url'),
            metadata.get('title'),
            metadata.get('summary'),
            metadata.get('content_type'),
            metadata.get('word_count'),
            datetime.now(),
            datetime.now()
        ))
        conn.commit()
        conn.close()
        
        # Store embeddings in Chroma
        chunk_ids = [f"{content_id}_chunk_{i}" for i in range(len(chunks))]
        chunk_metadata = [{"content_id": content_id, "chunk_index": i} for i in range(len(chunks))]
        
        self.collection.add(
            embeddings=embeddings,
            documents=chunks,
            metadatas=chunk_metadata,
            ids=chunk_ids
        )
    
    def search_content(self, query: str, n_results: int = 5) -> List[Dict]:
        """Search content using semantic similarity"""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        # Get content metadata for matching chunks
        content_ids = list(set([meta['content_id'] for meta in results['metadatas'][0]]))
        
        conn = sqlite3.connect(self.sqlite_path)
        cursor = conn.cursor()
        placeholders = ','.join(['?' for _ in content_ids])
        cursor.execute(f'''
            SELECT * FROM content_items WHERE id IN ({placeholders})
        ''', content_ids)
        
        metadata_results = cursor.fetchall()
        conn.close()
        
        return {
            'chunks': results['documents'][0],
            'distances': results['distances'][0],
            'metadata': metadata_results
        }
    
    def get_all_content(self) -> List[Dict]:
        """Get all content metadata"""
        conn = sqlite3.connect(self.sqlite_path)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM content_items ORDER BY created_at DESC')
        results = cursor.fetchall()
        conn.close()
        
        columns = ['id', 'url', 'title', 'summary', 'content_type', 'word_count', 'created_at', 'updated_at']
        return [dict(zip(columns, row)) for row in results]
