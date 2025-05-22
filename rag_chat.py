from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from typing import List, Dict, Tuple
import config
import os

class RAGChatSystem:
    def __init__(self, chroma_db_path: str):
        # Initialize LangChain components
        self.llm = ChatOpenAI(
            model_name=config.MODEL_NAME,
            openai_api_key=config.OPENAI_API_KEY,
            temperature=0.7
        )
        
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            openai_api_key=config.OPENAI_API_KEY
        )
        
        # Initialize memory for conversation
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        # Connect to existing Chroma database
        self.vectorstore = None
        self.retrieval_chain = None
        self._initialize_vectorstore(chroma_db_path)
    
    def _initialize_vectorstore(self, chroma_db_path: str):
        """Initialize connection to existing Chroma database"""
        try:
            if os.path.exists(chroma_db_path):
                # Connect to existing database
                self.vectorstore = Chroma(
                    persist_directory=chroma_db_path,
                    embedding_function=self.embeddings,
                    collection_name="content_inventory"
                )
                
                # Create retrieval chain
                retriever = self.vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 5}
                )
                
                # This is where LangChain really shines - sophisticated RAG chain
                self.retrieval_chain = ConversationalRetrievalChain.from_llm(
                    llm=self.llm,
                    retriever=retriever,
                    memory=self.memory,
                    return_source_documents=True,
                    verbose=False,
                    chain_type="stuff"
                )
            else:
                print(f"Vector database not found at {chroma_db_path}")
        except Exception as e:
            print(f"Error initializing vectorstore: {e}")
    
    def chat(self, question: str) -> Dict:
        """Chat with your content using RAG"""
        if not self.retrieval_chain:
            return {
                "answer": "Sorry, I don't have access to your content library yet. Please add some content first!",
                "source_documents": [],
                "success": False
            }
        
        try:
            # Use LangChain's conversational retrieval chain
            result = self.retrieval_chain({"question": question})
            
            return {
                "answer": result["answer"],
                "source_documents": result.get("source_documents", []),
                "success": True
            }
        except Exception as e:
            return {
                "answer": f"Sorry, I encountered an error: {str(e)}",
                "source_documents": [],
                "success": False
            }
    
    def get_conversation_history(self) -> List[Dict]:
        """Get the conversation history from memory"""
        if self.memory and hasattr(self.memory, 'chat_memory'):
            messages = []
            for message in self.memory.chat_memory.messages:
                messages.append({
                    "type": message.__class__.__name__,
                    "content": message.content
                })
            return messages
        return []
    
    def clear_memory(self):
        """Clear conversation memory"""
        if self.memory:
            self.memory.clear()
