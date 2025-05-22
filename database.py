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
        self.collection.add(
            ids=chunk_ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=[{"content_id": content_id} for _ in chunks]
        )