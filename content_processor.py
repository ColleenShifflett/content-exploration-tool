import hashlib
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import Document
from langchain.chains.summarize import load_summarize_chain
import config

class ContentProcessor:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            openai_api_key=config.OPENAI_API_KEY
        )
        self.llm = ChatOpenAI(
            model_name=config.MODEL_NAME,
            openai_api_key=config.OPENAI_API_KEY,
            temperature=0.3
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )
    
    def extract_from_url(self, url: str) -> Tuple[str, str, str]:
        """Extract content from a URL"""
        try:
            # Add headers to make the request look like it's from a browser
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title = soup.find('title')
            title = title.get_text().strip() if title else "Untitled"
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract text content
            content = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in content.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            content = ' '.join(chunk for chunk in chunks if chunk)
            
            # Limit content length for MVP
            if len(content) > config.MAX_CONTENT_LENGTH:
                content = content[:config.MAX_CONTENT_LENGTH] + "..."
            
            return title, content, "web_page"
            
        except requests.RequestException as e:
            raise Exception(f"Error fetching URL: {str(e)}")
        except Exception as e:
            raise Exception(f"Error extracting content from URL: {str(e)}")
    
    def process_text_content(self, text: str, title: str = "Text Document") -> Tuple[str, str, str]:
        """Process raw text content"""
        # Limit content length for MVP
        if len(text) > config.MAX_CONTENT_LENGTH:
            text = text[:config.MAX_CONTENT_LENGTH] + "..."
        
        return title, text, "text_document"
    
    def generate_summary(self, content: str) -> str:
        """Generate a summary of the content using LangChain"""
        try:
            docs = [Document(page_content=content)]
            summarize_chain = load_summarize_chain(self.llm, chain_type="stuff")
            summary = summarize_chain.run(docs)
            return summary.strip()
        except Exception as e:
            return f"Summary generation failed: {str(e)}"
    
    def create_chunks_and_embeddings(self, content: str) -> Tuple[List[str], List[List[float]]]:
        """Split content into chunks and create embeddings"""
        # Split content into chunks
        docs = self.text_splitter.create_documents([content])
        chunks = [doc.page_content for doc in docs]
        
        # Create embeddings
        embeddings = self.embeddings.embed_documents(chunks)
        
        return chunks, embeddings
    
    def generate_content_id(self, content: str, url: str = None) -> str:
        """Generate a unique ID for content"""
        source = url if url else content[:100]
        return hashlib.md5(source.encode()).hexdigest()
    
    def process_content(self, source: str, content_type: str = "url") -> Dict:
        """Main processing function"""
        try:
            # Extract content based on type
            if content_type == "url":
                title, content, content_type = self.extract_from_url(source)
                url = source
            else:  # text
                title, content, content_type = self.process_text_content(source)
                url = None
            
            # Generate content ID
            content_id = self.generate_content_id(content, url)
            
            # Generate summary
            summary = self.generate_summary(content)
            
            # Create chunks and embeddings
            chunks, embeddings = self.create_chunks_and_embeddings(content)
            
            # Prepare metadata
            metadata = {
                'url': url,
                'title': title,
                'summary': summary,
                'content_type': content_type,
                'word_count': len(content.split())
            }
            
            return {
                'content_id': content_id,
                'metadata': metadata,
                'chunks': chunks,
                'embeddings': embeddings,
                'success': True
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
