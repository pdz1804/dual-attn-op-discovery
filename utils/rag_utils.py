import logging
import os
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
import json
import pickle
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

from configs.paths import *
from configs.hyperparams import *

logger = logging.getLogger(__name__)

class RAGProcessor:
    """
    RAG (Retrieval-Augmented Generation) processor for company-patent matching using ChromaDB
    """
    
    def __init__(self, embedder, db_path=None):
        self.embedder = embedder
        self.db_path = db_path or RAG_VECTOR_DB_DIR
        self.client = None
        self.collection = None
        self.collection_name = "company_documents"
        
        # Ensure directory exists
        os.makedirs(self.db_path, exist_ok=True)
        
        # Initialize ChromaDB client
        self._initialize_chromadb()
    
    def _initialize_chromadb(self):
        """Initialize ChromaDB client and settings"""
        try:
            # Create ChromaDB client with persistent storage
            self.client = chromadb.PersistentClient(
                path=self.db_path,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            logger.info(f"ChromaDB client initialized at {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {str(e)}")
            raise
    
    def create_document_chunks(self, documents: List[Dict], chunk_size=RAG_CHUNK_SIZE, chunk_overlap=RAG_CHUNK_OVERLAP):
        """
        Create chunks from documents for better retrieval
        """
        chunks = []
        
        for doc in documents:
            content = doc['content']
            
            # For now, we'll treat each company's keywords as a single chunk
            # In the future, we could implement more sophisticated chunking
            chunks.append({
                'content': content,
                'company_id': doc['company_id'],
                'company_name': doc.get('company_name', 'Unknown'),
                'metadata': doc.get('metadata', {}),
                'chunk_id': f"{doc['company_id']}_0"
            })
        
        logger.info(f"Created {len(chunks)} document chunks")
        return chunks
    
    def build_vector_database(self, documents: List[Dict], force_rebuild=False):
        """
        Build ChromaDB vector database from documents
        """
        try:
            # Check if collection exists
            try:
                existing_collection = self.client.get_collection(name=self.collection_name)
                if not force_rebuild:
                    logger.info("Loading existing ChromaDB collection...")
                    self.collection = existing_collection
                    count = self.collection.count()
                    logger.info(f"Loaded existing collection with {count} documents")
                    return
                else:
                    # Delete existing collection for rebuild
                    self.client.delete_collection(name=self.collection_name)
                    logger.info("Deleted existing collection for rebuild")
            except Exception:
                # Collection doesn't exist, will create new one
                pass
            
            logger.info("Building new ChromaDB collection...")
            
            # Create chunks
            chunks = self.create_document_chunks(documents)
            
            # Create new collection
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Company documents for patent-product matching"}
            )
            
            # Prepare data for ChromaDB
            chunk_texts = [chunk['content'] for chunk in chunks]
            chunk_ids = [chunk['chunk_id'] for chunk in chunks]
            chunk_metadatas = [
                {
                    'company_id': chunk['company_id'],
                    'company_name': chunk['company_name'],
                    'source': chunk['metadata'].get('source', 'unknown'),
                    'keywords': str(chunk['metadata'].get('keywords', []))
                }
                for chunk in chunks
            ]
            
            # Generate embeddings using our embedder
            logger.info("Generating embeddings for document chunks...")
            embeddings = self.embedder.encode_texts(chunk_texts)
            
            # Convert numpy arrays to lists for ChromaDB
            embeddings_list = [emb.tolist() for emb in embeddings]
            
            # Add documents to collection in batches
            batch_size = 100
            for i in range(0, len(chunk_texts), batch_size):
                end_idx = min(i + batch_size, len(chunk_texts))
                
                self.collection.add(
                    embeddings=embeddings_list[i:end_idx],
                    documents=chunk_texts[i:end_idx],
                    metadatas=chunk_metadatas[i:end_idx],
                    ids=chunk_ids[i:end_idx]
                )
                
                logger.info(f"Added batch {i//batch_size + 1}/{(len(chunk_texts)-1)//batch_size + 1}")
            
            logger.info(f"ChromaDB collection built with {len(chunks)} documents")
            
        except Exception as e:
            logger.error(f"Error building vector database: {str(e)}")
            raise
    
    def search_similar_documents(self, query: str, top_k: int = RAG_TOP_K) -> List[Dict]:
        """
        Search for similar documents using the query
        """
        if self.collection is None:
            raise ValueError("Vector database not initialized. Please build or load first.")
        
        try:
            # Generate query embedding
            query_embedding = self.embedder.encode_text(query)
            
            # Search using ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Process results
            search_results = []
            if results['documents'] and len(results['documents']) > 0:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0]
                distances = results['distances'][0]
                
                for doc, metadata, distance in zip(documents, metadatas, distances):
                    # Convert distance to similarity score (ChromaDB uses L2 distance)
                    # Similarity = 1 / (1 + distance)
                    similarity_score = 1.0 / (1.0 + distance)
                    
                    # Parse keywords back to list
                    keywords = []
                    if metadata.get('keywords'):
                        try:
                            keywords = eval(metadata['keywords']) if metadata['keywords'].startswith('[') else []
                        except:
                            keywords = []
                    
                    search_results.append({
                        'score': similarity_score,
                        'company_id': metadata['company_id'],
                        'company_name': metadata['company_name'],
                        'content': doc,
                        'metadata': {
                            'source': metadata.get('source', 'unknown'),
                            'keywords': keywords
                        }
                    })
            
            return search_results
            
        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            return []
    
    def rag_query_companies(self, product_query: str, top_k: int = RAG_TOP_K) -> List[Dict]:
        """
        Use RAG to find relevant companies for a product query
        """
        logger.info(f"Processing RAG query: '{product_query}'")
        
        # Search for similar documents
        similar_docs = self.search_similar_documents(product_query, top_k)
        
        # Process results to get company information
        company_results = []
        for doc in similar_docs:
            company_results.append({
                'company_id': doc['company_id'],
                'company_name': doc['company_name'],
                'relevance_score': doc['score'],
                'matching_content': doc['content'],
                'keywords': doc['metadata'].get('keywords', []),
                'source': doc['metadata'].get('source', 'unknown')
            })
        
        logger.info(f"RAG query returned {len(company_results)} relevant companies")
        return company_results
    
    def get_collection_stats(self):
        """Get statistics about the current collection"""
        if self.collection is None:
            return {"status": "No collection loaded"}
        
        try:
            count = self.collection.count()
            return {
                "status": "Active",
                "document_count": count,
                "collection_name": self.collection_name
            }
        except Exception as e:
            return {"status": f"Error: {str(e)}"}

def create_rag_processor(embedder, force_rebuild=False):
    """
    Factory function to create and initialize RAG processor
    """
    rag_processor = RAGProcessor(embedder, db_path=RAG_VECTOR_DB_DIR)
    
    # Load company data for RAG
    if os.path.exists(EMBEDDINGS_OUTPUT):
        logger.info("Loading company keywords for RAG...")
        company_df = pd.read_csv(EMBEDDINGS_OUTPUT)
        
        # Create RAG documents
        from utils.vector_utils import create_rag_documents
        documents = create_rag_documents(
            company_df, 
            use_external_summaries=RAG_USE_EXTERNAL_SUMMARIES,
            external_summaries_path=RAG_EXTERNAL_SUMMARIES
        )
        
        # Build vector database
        rag_processor.build_vector_database(documents, force_rebuild=force_rebuild)
    else:
        logger.warning(f"Company embeddings file not found: {EMBEDDINGS_OUTPUT}")
    
    return rag_processor

def query_rag_pipeline(product_query: str, embedder, top_k: int = RAG_TOP_K) -> Dict:
    """
    Complete RAG pipeline for querying companies
    """
    # Create RAG processor
    rag_processor = create_rag_processor(embedder)
    
    # Query for relevant companies
    company_results = rag_processor.rag_query_companies(product_query, top_k)
    
    return {
        'query': product_query,
        'results': company_results,
        'total_results': len(company_results),
        'method': 'RAG (ChromaDB)'
    }

def combine_rag_with_dual_attention(product_query: str, embedder, dual_attention_results: List[Dict], top_k: int = RAG_TOP_K) -> Dict:
    """
    Combine RAG results with dual attention model results
    """
    # Get RAG results
    rag_results = query_rag_pipeline(product_query, embedder, top_k)
    
    # Create a combined scoring system
    combined_results = []
    
    # Create lookup for dual attention results
    da_lookup = {str(r.get('company_id', '')): r for r in dual_attention_results}
    
    for rag_result in rag_results['results']:
        company_id = str(rag_result['company_id'])
        
        # Start with RAG score
        combined_score = rag_result['relevance_score']
        
        # Add dual attention score if available
        if company_id in da_lookup:
            da_result = da_lookup[company_id]
            da_score = da_result.get('cosine_similarity', 0)
            # Weighted combination (adjust weights as needed)
            combined_score = 0.6 * combined_score + 0.4 * da_score
        
        combined_results.append({
            'company_id': company_id,
            'company_name': rag_result['company_name'],
            'combined_score': combined_score,
            'rag_score': rag_result['relevance_score'],
            'dual_attention_score': da_lookup.get(company_id, {}).get('cosine_similarity', 0),
            'rag_content': rag_result['matching_content'],
            'keywords': rag_result['keywords'],
            'has_dual_attention': company_id in da_lookup
        })
    
    # Sort by combined score
    combined_results.sort(key=lambda x: x['combined_score'], reverse=True)
    
    return {
        'query': product_query,
        'results': combined_results[:top_k],
        'total_results': len(combined_results),
        'method': 'RAG (ChromaDB) + Dual Attention'
    } 