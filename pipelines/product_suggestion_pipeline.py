"""
Product Suggestion Pipeline
===========================

A robust product suggestion system designed for integration with the FullFlow patent analysis pipeline.
Works with companies data from dual attention training and transformation matrix.

This pipeline:
- Takes companies with hojinid, keywords, and patents as input
- Suggests products based on user patent abstraction query
- Uses both company keywords and their patent data for suggestions
- Exports results to JSON format with timestamps
- Optionally enhances product names using OpenAI API

Dependencies: sentence-transformers, scikit-learn, numpy, tqdm, json, openai (optional)
"""

import os
import sys
import json
import logging
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Set, Any
import numpy as np
from tqdm import tqdm
from itertools import combinations
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from functools import lru_cache
import re
import warnings
from collections import Counter, defaultdict

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import configurations
from configs.hyperparams import *
from configs.paths import US_PATENT_DATA, BASE_DIR

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logger = logging.getLogger(__name__)

# === PRODUCT SUGGESTION CONFIGURATION ===

# Patent processing limits are now defined in configs/hyperparams.py

# Default product suggestion parameters
DEFAULT_CONFIG = {
    'model_name': 'all-MiniLM-L6-v2',
    'alpha': 0.6,
    'beta': 0.5,
    'similarity_threshold': 0.15,
    'top_k_suggestions': 3,
    'max_keywords': 20,
    'max_combinations': 4,
    'use_patent_data': True,
    'enable_openai_enhance': False,
    'openai_model': 'gpt-4o-mini',
    'output_directory': 'data/suggestions'
}

# Class-level cache for patent abstracts (shared across all instances)
_PATENT_ABSTRACT_CACHE = None
_PATENT_STATS = {'total_patents': 0, 'full_abstracts': 0, 'fallback_abstracts': 0}


class PatentProductSuggester:
    """
    Patent-based product suggestion system integrated with FullFlow pipeline.
    
    This system works with:
    1. Companies data from clustering/transformation matrix pipeline
    2. Company keywords from dual attention training
    3. Company patents with abstractions
    4. User patent abstraction queries
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the PatentProductSuggester.
        
        Args:
            config: Configuration dictionary with product suggestion parameters
        """
        # Merge with default config
        self.config = {**PRODUCT_SUGGESTION_CONFIG}
        if config:
            self.config.update(config)
        
        self.alpha = self.config['alpha']
        self.beta = self.config['beta']
        self.similarity_threshold = self.config['similarity_threshold']
        self.top_k_suggestions = self.config['top_k_suggestions']
        self.max_keywords = self.config['max_keywords']
        self.max_combinations = self.config['max_combinations']
        self.use_patent_data = self.config['use_patent_data']
        self.enable_openai_enhance = self.config['enable_openai_enhance']
        self.debug_enabled = self.config.get('debug_enabled', False)
        
        # Initialize models
        logger.info(f"Loading sentence embedding model: {self.config['model_name']}")
        self.embedder = SentenceTransformer(self.config['model_name'])
        # Disable progress bars for cleaner output
        self.embedder.encode("test", show_progress_bar=False)  # Initialize and set default
        
        # Initialize OpenAI if enabled
        self.openai_client = None
        if self.enable_openai_enhance:
            self._initialize_openai()
        
        # Cache for embeddings and computations
        self._embedding_cache = {}
        self._keyword_cache = {}
        
        # Load original patent data for full abstracts (cached globally)
        self._patent_abstract_mapping = self._load_patent_abstracts()
        
        # Display patent processing statistics
        self._show_patent_stats()
        
        # Ensure output directory exists
        os.makedirs(self.config['output_directory'], exist_ok=True)
        
        logger.info("PatentProductSuggester initialized successfully")
    
    def _initialize_openai(self):
        """Initialize OpenAI client if enabled and API key is available."""
        try:
            import openai
            from dotenv import load_dotenv
            
            # Load environment variables
            load_dotenv()
            
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                logger.warning("OpenAI API key not found in environment. Product enhancement disabled.")
                self.enable_openai_enhance = False
                return
            
            # Initialize OpenAI client
            self.openai_client = openai.OpenAI(api_key=api_key)
            logger.info("OpenAI client initialized successfully for product name enhancement")
        except Exception as e:
            logger.warning(f"Could not initialize OpenAI client: {e}")
            self.enable_openai_enhance = False
    
    def _load_patent_abstracts(self) -> Dict[str, str]:
        """
        Load original patent data and create mapping from patent ID to full abstract.
        
        Returns:
            Dictionary mapping patent IDs to full abstract texts
        """
        global _PATENT_ABSTRACT_CACHE, _PATENT_STATS
        if _PATENT_ABSTRACT_CACHE is not None:
            return _PATENT_ABSTRACT_CACHE

        try:
            logger.info(f"Loading original patent data from {US_PATENT_DATA}")
            patent_df = pd.read_csv(US_PATENT_DATA)
            
            # Create mapping from patent ID to abstract
            # The exact column names may vary, so try common variations
            patent_id_col = None
            abstract_col = None 
            
            # Find patent ID column
            possible_id_cols = ['appln_id', 'patpubnr', 'publn_nr', 'patent_id', 'id', 'patent_number', 'patentid', 'patent_no']
            for col in possible_id_cols:
                if col in patent_df.columns:
                    patent_id_col = col
                    break
            
            # Find abstract column  
            possible_abstract_cols = ['patent_abstract', 'abstract', 'Abstract', 'abstraction']
            for col in possible_abstract_cols:
                if col in patent_df.columns:
                    abstract_col = col
                    break
            
            if not patent_id_col or not abstract_col:
                logger.warning(f"Could not find patent ID or abstract columns in {US_PATENT_DATA}")
                logger.info(f"Available columns: {list(patent_df.columns)}")
                logger.info(f"Found patent ID column: {patent_id_col}")
                logger.info(f"Found abstract column: {abstract_col}")
                return {}
            
            logger.info(f"Using patent ID column: '{patent_id_col}' and abstract column: '{abstract_col}'")
            
            # Create mapping, handling missing values
            mapping = {}
            sample_patent_ids = []
            for _, row in patent_df.iterrows():
                patent_id = str(row[patent_id_col])
                abstract = row[abstract_col]
                if pd.notna(abstract) and str(abstract).strip():
                    mapping[patent_id] = str(abstract).strip()
                    _PATENT_STATS['full_abstracts'] += 1
                    # Collect first 5 patent IDs for debugging
                    if len(sample_patent_ids) < 5:
                        sample_patent_ids.append(patent_id)
                else:
                    _PATENT_STATS['fallback_abstracts'] += 1
                _PATENT_STATS['total_patents'] += 1
            
            _PATENT_ABSTRACT_CACHE = mapping
            logger.info(f"Loaded {len(mapping)} patent abstracts from original data")
            logger.info(f"Sample patent IDs from original file: {sample_patent_ids}")
            return mapping
            
        except Exception as e:
            logger.error(f"Error loading patent abstracts: {e}")
            return {}
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text for processing."""
        if not isinstance(text, str):
            return ""
        
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()
    
    def extract_keywords(self, text: str, top_k: Optional[int] = None) -> List[str]:
        """Extract keywords from text using TF-IDF."""
        if top_k is None:
            top_k = self.max_keywords
            
        cache_key = f"{hash(text)}_{top_k}"
        if cache_key in self._keyword_cache:
            return self._keyword_cache[cache_key]
        
        cleaned_text = self.clean_text(text)
        if not cleaned_text:
            return []
        
        try:
            vectorizer = TfidfVectorizer(
                stop_words='english',
                max_features=5000,
                ngram_range=(1, 1),  # Only use single words (1-grams) for better keyword matching
                min_df=1,
                max_df=1.0,
                lowercase=True,  # Ensure consistent case
                token_pattern=r'\b[a-zA-Z]{2,}\b'  # Only extract words with 2+ letters
            )
            
            tfidf_matrix = vectorizer.fit_transform([cleaned_text])
            feature_names = vectorizer.get_feature_names_out()
            tfidf_scores = tfidf_matrix.toarray()[0]
            
            keyword_scores = list(zip(feature_names, tfidf_scores))
            keyword_scores.sort(key=lambda x: x[1], reverse=True)
            
            keywords = [keyword for keyword, score in keyword_scores[:top_k] if score > 0]
            self._keyword_cache[cache_key] = keywords
            return keywords
            
        except Exception as e:
            logger.warning(f"Keyword extraction failed: {e}")
            # Fallback: split into individual words and filter
            words = cleaned_text.lower().split()
            words = [w.strip() for w in words if len(w.strip()) >= 2 and w.strip().isalpha()]
            return list(set(words))[:top_k]
    
    @lru_cache(maxsize=4096)
    def get_embedding(self, text: str) -> np.ndarray:
        """Get normalized sentence embedding with caching."""
        try:
            embedding = self.embedder.encode([text], convert_to_numpy=True, show_progress_bar=False)[0]
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            return embedding
        except Exception as e:
            logger.warning(f"Embedding generation failed: {e}")
            return np.zeros(self.embedder.get_sentence_embedding_dimension())
    
    def generate_product_names(
        self, 
        company_keywords: List[str], 
        patent_keywords: List[str] = None
    ) -> List[str]:
        """Generate domain-aware product names from company keywords and patent data."""
        
        # Combine keywords from multiple sources
        all_keywords = list(set(company_keywords))
        if patent_keywords and self.use_patent_data:
            # Merge and prioritize keywords
            keyword_freq = Counter(company_keywords + patent_keywords)
            # Take top keywords by frequency
            all_keywords = [kw for kw, _ in keyword_freq.most_common(self.max_keywords)]
        
        # FIXED: Filter out short keywords BEFORE combination generation to avoid single-word products
        valid_keywords = []
        for kw in all_keywords:
            cleaned_kw = kw.replace('_', ' ').strip()
            # Keep keywords that are at least 3 characters and contain only letters/spaces
            if len(cleaned_kw) > 2 and cleaned_kw.replace(' ', '').isalpha():
                valid_keywords.append(cleaned_kw)
        
        all_keywords = valid_keywords
        
        if self.debug_enabled:
            logger.info(f"Product generation debug: {len(valid_keywords)} valid keywords after filtering: {valid_keywords[:10]}")
            logger.info(f"Max combinations setting: {self.max_combinations}")
        
        if not all_keywords:
            return []
        
        # Detect domain based on keywords
        detected_domains = self._detect_domains(all_keywords)
        
        # Log detected domains for transparency
        if detected_domains:
            logger.info(f"Detected domains for product generation: {', '.join(detected_domains)}")
        
        # Domain-specific templates and terminology
        domain_templates = {
            'pharmaceutical': {
                'templates': [
                    "{} Inhibitor", "{} Therapy", "{} Treatment", "{} Drug",
                    "{} Compound", "{} Formula", "{} Therapeutic", "{} Medication",
                    "{} Derivative", "{} Agent", "{} Pharmaceutical", "{} Medicine"
                ],
                'combinations': [
                    "{} {} Therapy", "{} {} Treatment", "{} {} Drug",
                    "{} {} Compound", "{} {} Inhibitor", "{} {} Agent"
                ]
            },
            'medical': {
                'templates': [
                    "{} Device", "{} Scanner", "{} Monitor", "{} Analyzer",
                    "{} Diagnostic", "{} Imaging", "{} Equipment", "{} Instrument",
                    "{} System", "{} Tool", "{} Machine", "{} Detector"
                ],
                'combinations': [
                    "{} {} System", "{} {} Device", "{} {} Scanner",
                    "{} {} Monitor", "{} {} Analyzer", "{} {} Equipment"
                ]
            },
            'technology': {
                'templates': [
                    "{} Algorithm", "{} Software", "{} Platform", "{} Framework",
                    "{} API", "{} Engine", "{} System", "{} Application",
                    "{} Interface", "{} Protocol", "{} Network", "{} Architecture"
                ],
                'combinations': [
                    "{} {} Platform", "{} {} System", "{} {} Engine",
                    "{} {} Framework", "{} {} Application", "{} {} Network"
                ]
            },
            'manufacturing': {
                'templates': [
                    "{} Machine", "{} Equipment", "{} Process", "{} Assembly",
                    "{} Production", "{} Fabrication", "{} Manufacturing", "{} Machinery",
                    "{} Automation", "{} Tool", "{} Line", "{} Plant"
                ],
                'combinations': [
                    "{} {} Machine", "{} {} Process", "{} {} System",
                    "{} {} Equipment", "{} {} Line", "{} {} Plant"
                ]
            },
            'energy': {
                'templates': [
                    "{} Cell", "{} Battery", "{} Generator", "{} Panel",
                    "{} Reactor", "{} Turbine", "{} Engine", "{} Power",
                    "{} Energy", "{} Fuel", "{} Solar", "{} Wind"
                ],
                'combinations': [
                    "{} {} Cell", "{} {} Battery", "{} {} Generator",
                    "{} {} System", "{} {} Power", "{} {} Energy"
                ]
            },
            'materials': {
                'templates': [
                    "{} Material", "{} Composite", "{} Alloy", "{} Polymer",
                    "{} Coating", "{} Fiber", "{} Membrane", "{} Catalyst",
                    "{} Substrate", "{} Film", "{} Nanoparticle", "{} Crystal"
                ],
                'combinations': [
                    "{} {} Material", "{} {} Composite", "{} {} Coating",
                    "{} {} Polymer", "{} {} Alloy", "{} {} Membrane"
                ]
            },
            'automotive': {
                'templates': [
                    "{} Engine", "{} System", "{} Component", "{} Assembly",
                    "{} Control", "{} Sensor", "{} Module", "{} Unit",
                    "{} Drive", "{} Transmission", "{} Brake", "{} Suspension"
                ],
                'combinations': [
                    "{} {} System", "{} {} Engine", "{} {} Control",
                    "{} {} Component", "{} {} Module", "{} {} Assembly"
                ]
            },
            'electronics': {
                'templates': [
                    "{} Chip", "{} Circuit", "{} Processor", "{} Sensor",
                    "{} Display", "{} Component", "{} Module", "{} Board",
                    "{} Device", "{} Controller", "{} Interface", "{} Amplifier"
                ],
                'combinations': [
                    "{} {} Chip", "{} {} Circuit", "{} {} Processor",
                    "{} {} Sensor", "{} {} Module", "{} {} Controller"
                ]
            },
            'agricultural': {
                'templates': [
                    "{} Fertilizer", "{} Pesticide", "{} Seed", "{} Crop",
                    "{} Herbicide", "{} Growth", "{} Plant", "{} Soil",
                    "{} Nutrient", "{} Treatment", "{} Protection", "{} Enhancement"
                ],
                'combinations': [
                    "{} {} Treatment", "{} {} Protection", "{} {} Growth",
                    "{} {} Enhancement", "{} {} Fertilizer", "{} {} System"
                ]
            },
            'financial': {
                'templates': [
                    "{} Platform", "{} System", "{} Service", "{} Solution",
                    "{} Analytics", "{} Management", "{} Trading", "{} Payment",
                    "{} Investment", "{} Risk", "{} Portfolio", "{} Exchange"
                ],
                'combinations': [
                    "{} {} Platform", "{} {} System", "{} {} Service",
                    "{} {} Analytics", "{} {} Management", "{} {} Solution"
                ]
            },
            'generic': {
                'templates': [
                    "{}", "{} Pro", "{} Advanced", "{} Plus",
                    "{} System", "{} Platform", "{} Solution", "{} Technology",
                    "{} Service", "{} Tool", "{} Kit", "{} Suite"
                ],
                'combinations': [
                    "{} {}", "{} {} System", "{} {} Platform",
                    "{} {} Solution", "{} {} Technology", "{} {} Service"
                ]
            }
        }
        
        products = []
        
        # Generate products based on detected domains
        for domain in detected_domains:
            templates = domain_templates.get(domain, domain_templates['generic'])
            
            # Only combination products (2+ keywords) - no single keyword products
            if self.debug_enabled:
                logger.info(f"Generating combinations for domain '{domain}' with {len(all_keywords)} keywords, max_combinations={self.max_combinations}")
            
            for r in range(2, min(self.max_combinations + 1, len(all_keywords) + 1)):
                if self.debug_enabled:
                    logger.info(f"Processing {r}-keyword combinations")
                
                combinations_count = 0
                for combo in list(combinations(all_keywords, r))[:15]:  # Limit combinations per size
                    # FIXED: Keywords are already filtered, so just use them directly
                    combo_parts = [kw.strip() for kw in combo if kw.strip()]
                    
                    # FIXED: Ensure we always have exactly r keywords (no filtering that reduces count)
                    if len(combo_parts) == r and r >= 2:
                        combinations_count += 1
                        if self.debug_enabled and combinations_count <= 3:  # Log first few combinations
                            logger.info(f"  Valid {r}-combo #{combinations_count}: {combo_parts}")
                        if r == 2:
                            # Use domain-specific combination templates (expanded for more variety)
                            for template in templates['combinations'][:8]:  # Increased from 6 to 8
                                if template.count('{}') == 2:
                                    products.append(template.format(
                                        combo_parts[0].title(), 
                                        combo_parts[1].title()
                                    ))
                            
                            # Add additional 2-word combinations with basic templates
                            combo_str = " ".join(combo_parts).title()
                            products.extend([
                                f"{combo_str}",
                                f"{combo_str} Pro",
                                f"Advanced {combo_str}",
                                f"{combo_str} Suite"
                            ])
                        elif r == 3:
                            # Three-keyword combinations (expanded)
                            combo_str = " ".join(combo_parts).title()
                            products.extend([
                                f"{combo_str}",
                                f"{combo_str} System",
                                f"{combo_str} Platform",
                                f"Advanced {combo_str}",
                                f"Integrated {combo_str}",
                                f"{combo_str} Suite"
                            ])
                        else:
                            # Four+ keyword combinations
                            combo_str = " ".join(combo_parts).title()
                            products.extend([
                                f"{combo_str}",
                                f"{combo_str} System",
                                f"Advanced {combo_str}"
                            ])
        
        # Remove duplicates and return expanded set
        unique_products = list(set(products))
        
        if self.debug_enabled:
            logger.info(f"Generated {len(products)} total products, {len(unique_products)} unique products")
            logger.info(f"Sample products: {unique_products[:10]}")
        
        return unique_products[:300]  # Increased limit for multi-domain coverage
    
    def _detect_domains(self, keywords: List[str]) -> List[str]:
        """Detect domains based on keyword analysis."""
        
        # Domain keyword mappings
        domain_indicators = {
            'pharmaceutical': [
                'drug', 'medication', 'therapy', 'treatment', 'compound', 'molecule',
                'inhibitor', 'pharmaceutical', 'medicine', 'therapeutic', 'clinical',
                'dosage', 'formulation', 'antibody', 'protein', 'enzyme', 'receptor',
                'diabetes', 'cancer', 'disease', 'disorder', 'syndrome', 'pathology',
                'pharmacology', 'bioactive', 'derivative', 'analog', 'metabolite'
            ],
            'medical': [
                'medical', 'diagnostic', 'imaging', 'scanner', 'monitor', 'device',
                'equipment', 'instrument', 'healthcare', 'clinical', 'patient',
                'hospital', 'surgery', 'surgical', 'analysis', 'detection',
                'measurement', 'examination', 'screening', 'testing', 'probe'
            ],
            'technology': [
                'software', 'algorithm', 'computing', 'data', 'digital', 'network',
                'internet', 'web', 'application', 'platform', 'system', 'interface',
                'programming', 'code', 'database', 'server', 'cloud', 'ai',
                'machine', 'learning', 'neural', 'artificial', 'intelligence'
            ],
            'manufacturing': [
                'manufacturing', 'production', 'assembly', 'fabrication', 'process',
                'machinery', 'equipment', 'automation', 'industrial', 'factory',
                'plant', 'line', 'conveyor', 'robotic', 'cnc', 'welding',
                'cutting', 'molding', 'casting', 'forming', 'machining'
            ],
            'energy': [
                'energy', 'power', 'battery', 'solar', 'wind', 'fuel', 'cell',
                'generator', 'turbine', 'reactor', 'nuclear', 'renewable',
                'electricity', 'electrical', 'grid', 'storage', 'conversion',
                'efficiency', 'renewable', 'sustainable', 'green', 'clean'
            ],
            'materials': [
                'material', 'polymer', 'composite', 'alloy', 'metal', 'ceramic',
                'coating', 'surface', 'membrane', 'fiber', 'nanoparticle',
                'crystal', 'structure', 'properties', 'strength', 'durability',
                'corrosion', 'wear', 'thermal', 'mechanical', 'chemical'
            ],
            'automotive': [
                'automotive', 'vehicle', 'car', 'engine', 'motor', 'transmission',
                'brake', 'suspension', 'steering', 'tire', 'fuel', 'exhaust',
                'emission', 'safety', 'airbag', 'sensor', 'control', 'electric',
                'hybrid', 'autonomous', 'driving', 'navigation'
            ],
            'electronics': [
                'electronic', 'circuit', 'chip', 'processor', 'semiconductor',
                'transistor', 'diode', 'capacitor', 'resistor', 'sensor',
                'display', 'led', 'lcd', 'oled', 'microcontroller', 'fpga',
                'pcb', 'integrated', 'amplifier', 'oscillator', 'filter'
            ],
            'agricultural': [
                'agricultural', 'farming', 'crop', 'plant', 'seed', 'soil',
                'fertilizer', 'pesticide', 'herbicide', 'irrigation', 'harvest',
                'growth', 'cultivation', 'greenhouse', 'organic', 'sustainable',
                'nutrition', 'yield', 'protection', 'treatment', 'enhancement'
            ],
            'financial': [
                'financial', 'banking', 'investment', 'trading', 'payment',
                'transaction', 'money', 'currency', 'exchange', 'market',
                'portfolio', 'risk', 'insurance', 'loan', 'credit', 'debt',
                'finance', 'accounting', 'audit', 'compliance', 'regulation'
            ]
        }
        
        detected = set()
        keyword_text = ' '.join(keywords).lower()
        
        # Score each domain based on keyword matches
        domain_scores = {}
        for domain, indicators in domain_indicators.items():
            score = sum(1 for indicator in indicators if indicator in keyword_text)
            if score > 0:
                domain_scores[domain] = score
        
        # Select top domains (at least 1, maximum 3)
        if domain_scores:
            # Sort by score and take top domains
            sorted_domains = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)
            detected.update([domain for domain, score in sorted_domains[:3]])
        
        # Always include generic as fallback
        if not detected:
            detected.add('generic')
        
        return list(detected)
    
    def enhance_product_names_with_openai(
        self, 
        products: List[str], 
        company_name: str,
        company_keywords: List[str],
        user_query: str
    ) -> Dict[str, str]:
        """Enhance product names using OpenAI to make them more professional.
        
        Returns:
            Dict mapping original product names to enhanced names.
            If enhancement fails or is disabled, returns identity mapping.
        """
        
        # Create identity mapping if enhancement is disabled
        if not self.enable_openai_enhance or not self.openai_client:
            return {product: product for product in products}
        
        try:
            # Take only top products to avoid token limits
            top_products = products[:5]
            # top_products = products
            
            prompt = f"""
            Company: {company_name}
            Company Keywords: {', '.join(company_keywords[:10])}
            User Query: {user_query[:300]}
            
            Current Product Names:
            {chr(30).join(f"- {p}" for p in top_products)}
            
            Please improve these product names to be more professional, marketable, and relevant to the company's focus. 
            Keep the core concepts but make them sound more like real commercial products.
            Return exactly {len(top_products)} improved names, one per line, without bullet points or numbering.
            """
            
            response = self.openai_client.chat.completions.create(
                model=self.config['openai_model'],
                messages=[
                    {"role": "system", "content": "You are a product naming expert. Create professional, marketable product names."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=512,
                temperature=0
            )
            
            enhanced_names = response.choices[0].message.content.strip().split('\n')
            enhanced_names = [name.strip().strip('-').strip() for name in enhanced_names if name.strip()]
            
            # Ensure we have the same number of names
            if len(enhanced_names) == len(top_products):
                logger.info(f"Enhanced {len(enhanced_names)} product names using OpenAI")
                # Create mapping from original to enhanced names
                enhancement_mapping = {orig: enhanced for orig, enhanced in zip(top_products, enhanced_names)}
                # Add identity mapping for remaining products
                for product in products[len(top_products):]:
                    enhancement_mapping[product] = product
                return enhancement_mapping
            else:
                logger.warning(f"OpenAI returned {len(enhanced_names)} names but expected {len(top_products)}, using originals")
                return {product: product for product in products}
                
        except Exception as e:
            logger.warning(f"OpenAI enhancement failed: {e}")
            return {product: product for product in products}
    
    def compute_similarity(self, text1: str, text2: str) -> Dict[str, float]:
        """Compute comprehensive similarity metrics."""
        
        # Extract keywords for lexical similarity
        kw1 = set(self.extract_keywords(text1, KEYWORD_EXTRACTION_LIMITS['similarity_computation']))
        kw2 = set(self.extract_keywords(text2, KEYWORD_EXTRACTION_LIMITS['similarity_computation']))
        
        # Lexical similarity (Jaccard)
        intersection = kw1 & kw2
        union = kw1 | kw2
        lexical_sim = len(intersection) / len(union) if union else 0.0
        
        # Semantic similarity
        emb1 = self.get_embedding(self.clean_text(text1))
        emb2 = self.get_embedding(self.clean_text(text2))
        semantic_sim = max(0.0, min(1.0, np.dot(emb1, emb2)))
        
        # Combined similarity
        combined_sim = self.beta * lexical_sim + (1 - self.beta) * semantic_sim
        
        return {
            'lexical': lexical_sim,
            'semantic': semantic_sim,
            'combined': combined_sim
        }
    
    def analyze_company_patents(self, patents: List[Dict]) -> Dict[str, Any]:
        """Analyze company patents to extract insights."""
        if not patents:
            return {'keywords': [], 'themes': [], 'avg_similarity': 0.0}
        
        # Limit to top patents for analysis (same as similarity computation)
        max_patents_for_analysis = MAX_PATENTS_FOR_ANALYSIS
        patents_to_analyze = patents[:max_patents_for_analysis]
        
        # Extract keywords from all patents
        all_keywords = []
        patent_texts = []
        full_count = 0
        fallback_count = 0
        
        for patent in patents_to_analyze:
            # Get original full abstract from mapping
            patent_id = str(patent.get('id', patent.get('patent_id', '')))
            full_abstract = self._patent_abstract_mapping.get(patent_id, '')
            
            if full_abstract:
                # Use full original abstract
                patent_texts.append(full_abstract)
                keywords = self.extract_keywords(full_abstract, KEYWORD_EXTRACTION_LIMITS['patent_analysis'])
                all_keywords.extend(keywords)
                full_count += 1
            else:
                # Debug: Log what patent ID we're looking for
                if len(patents_to_analyze) <= 5:  # Only log for small patent sets to avoid spam
                    logger.info(f"Debug - Patent ID '{patent_id}' not found in original data mapping")
                
                # Fallback to tokenized abstract if original not found
                fallback_abstract = patent.get('abstract', '')
                if fallback_abstract:
                    # Handle both string and list formats
                    if isinstance(fallback_abstract, list):
                        fallback_abstract = ' '.join(fallback_abstract)
                    fallback_abstract = str(fallback_abstract)
                    patent_texts.append(fallback_abstract)
                    keywords = self.extract_keywords(fallback_abstract, KEYWORD_EXTRACTION_LIMITS['patent_analysis'])
                    all_keywords.extend(keywords)
                    fallback_count += 1
        
        # Log processing statistics
        # logger.info(f"Patent analysis: {full_count}/{len(patents)} full abstracts, {fallback_count}/{len(patents)} fallback")
        
        # Get most common keywords
        keyword_freq = Counter(all_keywords)
        top_keywords = [kw for kw, _ in keyword_freq.most_common(KEYWORD_EXTRACTION_LIMITS['top_keywords_limit'])]
        
        # Find common themes (keyword combinations that appear frequently)
        themes = []
        if len(patent_texts) > 1:
            # Simple theme detection based on co-occurring keywords
            theme_counter = Counter()
            for keywords in [self.extract_keywords(text, KEYWORD_EXTRACTION_LIMITS['theme_extraction']) for text in patent_texts]:
                for combo in combinations(keywords, 2):
                    theme_counter[combo] += 1
            
            themes = [' '.join(combo) for combo, count in theme_counter.most_common(KEYWORD_EXTRACTION_LIMITS['theme_combinations']) if count > 1]
        
        return {
            'keywords': top_keywords,
            'themes': themes,
            'patent_count': len(patents_to_analyze),  # Patents actually analyzed (limited)
            'total_patents': len(patents),  # Total patents available
            'full_abstract_count': full_count,
            'fallback_count': fallback_count,
            'avg_similarity': 0.0  # Could be computed if needed
        }
    
    def suggest_products_for_company(
        self, 
        user_query: str,
        company_data: Dict[str, Any],
        debug_enabled: bool = None
    ) -> Optional[Dict[str, Any]]:
        """Suggest products for a single company."""
        
        # Use instance debug setting if not explicitly provided
        if debug_enabled is None:
            debug_enabled = self.debug_enabled
        
        hojinid = company_data.get('hojinid', company_data.get('hojin_id', 'unknown'))
        company_name = company_data.get('name', company_data.get('company_name', f'Company_{hojinid}'))
        company_keywords = company_data.get('keywords', [])
        company_patents = company_data.get('patents', [])
        
        if not company_keywords and not company_patents:
            logger.warning(f"No data available for company {company_name}")
            return None
        
        # Analyze company patents
        patent_analysis = self.analyze_company_patents(company_patents)
        patent_keywords = patent_analysis['keywords']
        
        # Log patent processing statistics
        if company_patents:
            total_patents = len(company_patents)
            patents_to_process = min(total_patents, MAX_PATENTS_PER_COMPANY)  # We process up to 20 patents
            logger.info(f"Patent Processing - {company_name}: Processing {patents_to_process}/{total_patents} patents (limited to top-{MAX_PATENTS_PER_COMPANY})")
            logger.info(f"Patent Processing - {company_name}: {patent_analysis['full_abstract_count']}/{patent_analysis['patent_count']} patents using full abstracts")
        
        # Debug: Show user query keywords extraction (first company only to avoid spam)
        if debug_enabled and company_name and len(user_query.strip()) > 0:
            raw_user_keywords = self.extract_keywords(user_query)
            if len(raw_user_keywords) == 0:
                logger.info(f"Debug - No keywords extracted from user query: '{user_query[:100]}...'")
            else:
                logger.info(f"Debug - Extracted {len(raw_user_keywords)} keywords from user query: {raw_user_keywords[:10]}")
        
        # Compute similarities
        # 1. Keyword-based similarity with normalization
        user_keywords = set(self.extract_keywords(user_query))
        # Normalize keywords to lowercase and remove extra spaces
        user_keywords = {kw.lower().strip() for kw in user_keywords if kw.strip()}
        
        company_kw_set = set(company_keywords)
        # Normalize company keywords to lowercase and remove extra spaces  
        company_kw_set = {kw.lower().strip() for kw in company_kw_set if kw.strip()}
        
        patent_kw_set = set(patent_keywords)
        # Normalize patent keywords to lowercase and remove extra spaces
        patent_kw_set = {kw.lower().strip() for kw in patent_kw_set if kw.strip()}
        
        # Compute Jaccard similarity with normalized keywords
        combined_kw_set = company_kw_set | patent_kw_set
        intersection = user_keywords & combined_kw_set
        union = user_keywords | combined_kw_set
        kw_similarity = len(intersection) / len(union) if union else 0.0
        
        # Debug keyword similarity computation (only when debug enabled)
        if debug_enabled and len(user_keywords) > 0 and len(combined_kw_set) > 0:
            sample_user_kw = list(user_keywords)[:5]
            sample_company_kw = list(combined_kw_set)[:5]
            overlapping_kw = list(intersection)[:5]
            
            logger.info(f"Debug Keyword Similarity - {company_name}:")
            logger.info(f"  User keywords (sample): {sample_user_kw}")
            logger.info(f"  Company keywords (sample): {sample_company_kw}")
            logger.info(f"  Overlapping keywords: {overlapping_kw}")
            logger.info(f"  Intersection size: {len(intersection)}, Union size: {len(union)}")
            logger.info(f"  Keyword similarity: {kw_similarity:.4f}")
        
        # 2. Patent-based similarity
        patent_similarity = 0.0
        patent_full_count = 0
        patent_fallback_count = 0
        
        if company_patents and self.use_patent_data:
            patent_sims = []
            patents_processed = 0
            max_patents_to_process = MAX_PATENTS_PER_COMPANY  # Use configuration constant
            
            for patent in company_patents[:max_patents_to_process]:  # Only process top patents
                # Get original full abstract from mapping
                patent_id = str(patent.get('id', patent.get('patent_id', '')))
                full_abstract = self._patent_abstract_mapping.get(patent_id, '')
                
                if full_abstract:
                    # Use full original abstract
                    sim = self.compute_similarity(user_query, full_abstract)
                    patent_sims.append(sim['combined'])
                    patent_full_count += 1
                    patents_processed += 1
                else:
                    # Debug: Log what patent ID we're looking for (first 3 only to avoid spam)
                    if patent_full_count + patent_fallback_count < 3:
                        logger.info(f"Debug - Patent ID '{patent_id}' not found in original data mapping for similarity computation")
                    
                    # Fallback to tokenized abstract if original not found
                    fallback_abstract = patent.get('abstract', '')
                    if fallback_abstract:
                        # Handle both string and list formats
                        if isinstance(fallback_abstract, list):
                            fallback_abstract = ' '.join(fallback_abstract)
                        sim = self.compute_similarity(user_query, str(fallback_abstract))
                        patent_sims.append(sim['combined'])
                        patent_fallback_count += 1
                        patents_processed += 1
            
            patent_similarity = max(patent_sims) if patent_sims else 0.0
            
            # Log similarity processing stats
            if patents_processed > 0:
                logger.info(f"Similarity Computation - {company_name}: Used {patents_processed}/{len(company_patents)} patents for similarity calculation")
        
        # 3. Combined company similarity
        company_similarity = self.alpha * kw_similarity + (1 - self.alpha) * patent_similarity
        
        if company_similarity < self.similarity_threshold:
            return None
        
        # Generate product candidates
        product_candidates = self.generate_product_names(company_keywords, patent_keywords)
        
        if not product_candidates:
            return None
        
        # Enhance product names with OpenAI if enabled
        enhancement_mapping = {}
        if self.enable_openai_enhance:
            enhancement_mapping = self.enhance_product_names_with_openai(
                product_candidates, company_name, company_keywords, user_query
            )
        else:
            # Create identity mapping if enhancement is disabled
            enhancement_mapping = {product: product for product in product_candidates}
        
        # Score and rank product candidates (use enhanced names for scoring)
        scored_products = []
        for original_name in product_candidates:
            enhanced_name = enhancement_mapping.get(original_name, original_name)
            # Score based on enhanced name for better relevance
            product_score = self.compute_similarity(user_query, enhanced_name)
            
            product_data = {
                'product_name': enhanced_name,
                'score': product_score['combined'],
                'lexical_similarity': product_score['lexical'],
                'semantic_similarity': product_score['semantic']
            }
            
            # Add original name if different from enhanced (when OpenAI is used)
            if self.enable_openai_enhance and original_name != enhanced_name:
                product_data['original_name'] = original_name
            
            scored_products.append(product_data)
        
        # Sort by score and take top k
        scored_products.sort(key=lambda x: x['score'], reverse=True)
        top_products = scored_products[:self.top_k_suggestions]
        
        return {
            'company_id': hojinid,
            'company_name': company_name,
            'company_similarity': company_similarity,
            'keyword_similarity': kw_similarity,
            'patent_similarity': patent_similarity,
            'source': company_data.get('source', 'unknown'),  # Source: transformation_matrix or nearest_cluster
            'patent_stats': {
                'total_patents': len(company_patents),
                'full_abstracts_used': patent_full_count,
                'fallback_abstracts_used': patent_fallback_count
            },
            'products': top_products,
            'metadata': {
                'keywords_count': len(company_keywords),
                'patent_keywords_count': len(patent_keywords),
                'total_candidates_generated': len(product_candidates)
            }
        }
    
    def suggest_products(
        self, 
        user_query: str,
        companies_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Main method to suggest products for all companies."""
        
        logger.info(f"Processing product suggestions for {len(companies_data)} companies")
        logger.info(f"User query length: {len(user_query)} characters")
        
        if not user_query or not companies_data:
            return {'error': 'Invalid input data', 'results': []}
        
        all_results = []
        debug_counter = 0  # Counter to limit debugging output
        
        # Process each company
        for company_data in tqdm(companies_data, desc="Processing companies"):
            try:
                company_result = self.suggest_products_for_company(user_query, company_data, False)
                if company_result:
                    all_results.append(company_result)
                debug_counter += 1
            except Exception as e:
                company_name = company_data.get('name', company_data.get('company_name', 'unknown'))
                logger.error(f"Error processing company {company_name}: {e}")
                debug_counter += 1
                continue
        
        # Sort all results by company similarity
        all_results.sort(key=lambda x: x['company_similarity'], reverse=True)
        
        # Return comprehensive results
        return {
            'query': user_query,  # Add the query to results
            'timestamp': datetime.now().isoformat(),
            'total_companies_processed': len(companies_data),
            'companies_with_suggestions': len(all_results),
            'results': all_results,
            'summary': {
                'average_similarity': np.mean([r['company_similarity'] for r in all_results]) if all_results else 0.0,
                'max_similarity': max([r['company_similarity'] for r in all_results]) if all_results else 0.0,
                'min_similarity': min([r['company_similarity'] for r in all_results]) if all_results else 0.0,
                'total_products_suggested': sum(len(r['products']) for r in all_results)
            }
        }
    
    def convert_numpy_types(self, obj):
        """Convert numpy types to Python native types for JSON serialization."""
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)  
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self.convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    def export_results(self, results: Dict[str, Any], output_path: Optional[str] = None) -> str:
        """
        Export product suggestion results to JSON and text files.
        
        Args:
            results: Results dictionary from suggest_products
            output_path: Optional custom output path (for JSON)
            
        Returns:
            Path to the exported JSON file
        """
        # Export JSON file
        json_path = self._export_json(results, output_path)
        
        # Export text file
        text_path = self.export_results_as_text(results)
        
        logger.info(f"Results exported to JSON: {json_path}")
        logger.info(f"Results exported to text: {text_path}")
        
        return json_path
    
    def _export_json(self, results: Dict[str, Any], output_path: Optional[str] = None) -> str:
        """Export results as JSON file (original functionality)."""
        try:
            # Ensure output directory exists
            os.makedirs(self.config['output_directory'], exist_ok=True)
            
            # Generate timestamped filename if not provided
            if not output_path:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"product_suggestions_{timestamp}.json"
                output_path = os.path.join(self.config['output_directory'], filename)
            
            # Convert numpy types for JSON serialization
            json_results = self.convert_numpy_types(results)
            
            # Export to JSON
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(json_results, f, indent=2, ensure_ascii=False)
            
            # FIXED: Normalize path separators for consistent display
            return output_path.replace('\\', '/')
            
        except Exception as e:
            logger.error(f"Error exporting JSON results: {e}")
            return ""

    def export_results_as_text(self, results: Dict[str, Any], output_path: Optional[str] = None) -> str:
        """
        Export product suggestion results as a nicely formatted text file.
        
        Args:
            results: Results dictionary from suggest_products
            output_path: Optional custom output path
            
        Returns:
            Path to the exported text file
        """
        try:
            # Create submissions directory  
            submissions_dir = os.path.join(BASE_DIR, "data", "submissions")
            os.makedirs(submissions_dir, exist_ok=True)
            
            # Generate timestamped filename if not provided
            if not output_path:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"product_suggestions_{timestamp}.txt"
                # FIXED: Return relative path for consistency with JSON export
                relative_path = "data/submissions/" + filename  # Use forward slashes
                output_path = os.path.join(submissions_dir, filename)  # Full path for writing
                return_path = relative_path  # Relative path for display
            else:
                return_path = output_path
            
            # Create formatted text content
            content_lines = []
            
            # Header
            content_lines.append("=" * 80)
            content_lines.append("PRODUCT SUGGESTION RESULTS")
            content_lines.append("=" * 80)
            content_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            content_lines.append(f"Query: {results.get('query', 'N/A')}")
            content_lines.append(f"Total Companies: {len(results.get('results', []))}")
            content_lines.append(f"Companies with Suggestions: {len([r for r in results.get('results', []) if r.get('products', [])])}")
            content_lines.append("")
            
            # Company results
            for i, company_result in enumerate(results.get('results', []), 1):
                company_name = company_result.get('company_name', 'Unknown Company')
                company_id = company_result.get('company_id', company_result.get('hojinid', 'N/A'))
                products = company_result.get('products', [])
                
                # Company header
                source = company_result.get('source', 'unknown')
                source_display = "Transformation Matrix Top-K" if source == 'transformation_matrix' else \
                                "Nearest Cluster" if source == 'nearest_cluster' else source.title()
                
                content_lines.append(f"RANK {i}: {company_name}")
                content_lines.append("-" * 60)
                content_lines.append(f"Company ID: {company_id}")
                content_lines.append(f"Source: {source_display}")
                content_lines.append(f"Overall Similarity Score: {company_result.get('company_similarity', 0.0):.4f}")
                content_lines.append(f"Keyword Similarity: {company_result.get('keyword_similarity', 0.0):.4f}")
                content_lines.append(f"Patent Similarity: {company_result.get('patent_similarity', 0.0):.4f}")
                
                # Patent processing stats
                if 'patent_stats' in company_result:
                    stats = company_result['patent_stats']
                    content_lines.append(f"Patent Processing: {stats.get('full_abstracts_used', 0)}/{stats.get('total_patents', 0)} full abstracts used")
                
                # Products
                if products:
                    content_lines.append(f"\nSUGGESTED PRODUCTS ({len(products)} products):")
                    for j, product in enumerate(products, 1):
                        product_name = product.get('product_name', 'Unknown Product')
                        original_name = product.get('original_name')
                        score = product.get('score', 0.0)
                        lexical = product.get('lexical_similarity', 0.0)
                        semantic = product.get('semantic_similarity', 0.0)
                        
                        # Display product name(s)
                        if original_name and original_name != product_name:
                            # Show both original and enhanced names
                            content_lines.append(f"  {j}. {product_name}")
                            content_lines.append(f"     (Original: {original_name})")
                        else:
                            # Show only the product name
                            content_lines.append(f"  {j}. {product_name}")
                        
                        content_lines.append(f"     Overall Score: {score:.4f}")
                        content_lines.append(f"     Lexical Similarity: {lexical:.4f}")
                        content_lines.append(f"     Semantic Similarity: {semantic:.4f}")
                else:
                    content_lines.append("\nSUGGESTED PRODUCTS: None (below similarity threshold)")
                
                content_lines.append("")
            
            # Summary statistics
            if results.get('results'):
                all_scores = [r.get('company_similarity', 0) for r in results['results']]
                avg_score = sum(all_scores) / len(all_scores)
                max_score = max(all_scores)
                min_score = min(all_scores)
                
                content_lines.append("=" * 80)
                content_lines.append("SUMMARY STATISTICS")
                content_lines.append("=" * 80)
                content_lines.append(f"Average Company Similarity: {avg_score:.4f}")
                content_lines.append(f"Highest Company Similarity: {max_score:.4f}")
                content_lines.append(f"Lowest Company Similarity: {min_score:.4f}")
                
                # Product count statistics
                product_counts = [len(r.get('products', [])) for r in results['results']]
                total_products = sum(product_counts)
                content_lines.append(f"Total Products Suggested: {total_products}")
                if product_counts:
                    avg_products = total_products / len([c for c in product_counts if c > 0]) if any(product_counts) else 0
                    content_lines.append(f"Average Products per Company (with suggestions): {avg_products:.1f}")
            
            content_lines.append("")
            content_lines.append("=" * 80)
            content_lines.append("END OF REPORT")
            content_lines.append("=" * 80)
            
            # Write to file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(content_lines))
            
            logger.info(f"Text results exported to {output_path}")
            return return_path  # FIXED: Return relative path for UI consistency
            
        except Exception as e:
            logger.error(f"Error exporting text results: {e}")
            return ""

    def _show_patent_stats(self):
        """Display statistics about the loaded patent abstracts."""
        logger.info(f"Patent Abstract Statistics:")
        logger.info(f"Total patents loaded: {_PATENT_STATS['total_patents']}")
        logger.info(f"Full abstracts: {_PATENT_STATS['full_abstracts']}")
        logger.info(f"Fallback abstracts (missing or empty): {_PATENT_STATS['fallback_abstracts']}")
        logger.info(f"Unique abstracts loaded for mapping: {len(_PATENT_ABSTRACT_CACHE)}")
        
        # Show a sample comparison if data is available
        if _PATENT_ABSTRACT_CACHE:
            self._show_sample_abstract_comparison()
    
    def _show_sample_abstract_comparison(self):
        """Show a sample comparison between full and tokenized abstracts."""
        try:
            # Get a random patent ID from our cache
            import random
            sample_patent_id = random.choice(list(_PATENT_ABSTRACT_CACHE.keys()))
            full_abstract = _PATENT_ABSTRACT_CACHE[sample_patent_id]
            
            # Show comparison
            logger.info(f"Sample Abstract Verification (Patent {sample_patent_id}):")
            logger.info(f"Full Abstract: {full_abstract[:200]}{'...' if len(full_abstract) > 200 else ''}")
            
            # Simulate tokenized version
            tokenized_version = ' '.join(full_abstract.lower().split()[:15]) + "..."
            logger.info(f"Tokenized Version: {tokenized_version}")
            
        except Exception as e:
            logger.warning(f"Could not show sample abstract comparison: {e}")
    
    def _log_patent_processing_stats(self, company_name: str, patents: List[Dict], full_count: int, fallback_count: int):
        """Log statistics about patent processing for a company."""
        total = len(patents)
        if total > 0:
            logger.info(f"Patent Processing - {company_name}: {full_count}/{total} patents using full abstracts, {fallback_count}/{total} using fallback")


def convert_pipeline_results_to_suggestions_format(results: List[Dict]) -> List[Dict]:
    """
    Convert pipeline results to the format expected by product suggestions.
    
    Args:
        results: List of company results from patent_product_pipeline
        
    Returns:
        List of companies in the format expected by PatentProductSuggester
    """
    suggestions_format = []
    
    for result in results:
        # Extract company information
        company_data = {
            'hojinid': result.get('firm_id', result.get('hojin_id', 'unknown')),
            'name': result.get('company_name', result.get('firm_name', 'Unknown Company')),
            'keywords': result.get('keywords', []),
            'patents': []
        }
        
        # Convert patent information if available
        patent_ids = result.get('patent_ids', [])
        patent_texts = result.get('patent_texts', {})
        
        for patent_id in patent_ids[:10]:  # Limit to top 10 patents
            if patent_id in patent_texts:
                company_data['patents'].append({
                    'patent_id': patent_id,
                    'abstract': patent_texts[patent_id]
                })
        
        suggestions_format.append(company_data)
    
    return suggestions_format


def product_suggestion_pipeline(
    user_query: str,
    companies_data: List[Dict[str, Any]],
    config: Optional[Dict] = None
) -> str:
    """
    Main product suggestion pipeline function.
    
    Args:
        user_query: User's patent abstract or product query
        companies_data: List of companies with their data
        config: Configuration dictionary
        
    Returns:
        Path to the exported results file
    """
    logger.info("Starting product suggestion pipeline")
    
    # Initialize suggester with config
    suggester = PatentProductSuggester(config)
    
    # Generate suggestions
    results = suggester.suggest_products(user_query, companies_data)
    
    # Export results
    output_path = suggester.export_results(results)
    
    logger.info(f"Product suggestion pipeline completed. Results saved to: {output_path}")
    return output_path 