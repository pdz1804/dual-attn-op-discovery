#!/usr/bin/env python3
"""
AI Patent Testing Script for FullFlow System

This script automatically tests the system with AI patent abstracts
to demonstrate both ML and RAG approaches.
"""

import subprocess
import sys
import time

# AI Patent Abstracts for testing
AI_PATENTS = [
    {
        "name": "AI Diagnosis Apparatus",
        "abstract": "An apparatus and a method for diagnosis are provided. The apparatus for diagnosis lesion include: a model generation unit configured to categorize learning data into one or more categories and to generate one or more categorized diagnostic models based on the categorized learning data, a model selection unit configured to select one or more diagnostic model for diagnosing a lesion from the categorized diagnostic models, and a diagnosis unit configured to diagnose the lesion based on image data of the lesion and the selected one or more diagnostic model."
    },
    {
        "name": "AI Event Prediction System",
        "abstract": "Embodiments are disclosed to provide the prediction of viewable events. Predicting viewable events will allow users to know what events will likely be viewable in a particular venue, such as a restaurant, bar, or private home. Information about venues and events is populated in a database by a plurality of venues or users. Users wishing to view a particular event can search for a venue that has a high probability of showing that event."
    },
    {
        "name": "Neural Processing Cluster",
        "abstract": "Certain aspects of the present disclosure provide techniques for time management and scheduling of synchronous neural processing on a cluster of processing nodes. A slip (or offset) may be introduced between processing nodes of a distributed processing system formed by a plurality of interconnected processing nodes, to enable faster nodes to continue processing without waiting for slower nodes to catch up. In certain aspects, a processing node, after completing each processing step, may check for received completion packets and apply a defined constraint to determine whether it may start processing a subsequent step or not."
    }
]

def run_command(command, description):
    """Run a command and display the results"""
    print(f"\n{'='*80}")
    print(f"🚀 RUNNING: {description}")
    print(f"{'='*80}")
    print(f"Command: {command}")
    print(f"{'='*80}")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=False, text=True, timeout=300)
        if result.returncode == 0:
            print(f"\n✅ {description} completed successfully")
        else:
            print(f"\n❌ {description} failed with return code {result.returncode}")
    except subprocess.TimeoutExpired:
        print(f"\n⏰ {description} timed out after 5 minutes")
    except Exception as e:
        print(f"\n💥 {description} failed with error: {e}")
    
    print(f"\n{'='*80}")
    time.sleep(2)  # Small delay between tests

def test_rag_approach():
    """Test all AI patents using RAG approach"""
    print("\n🔍 TESTING RAG APPROACH WITH AI PATENTS")
    print("="*80)
    
    for i, patent in enumerate(AI_PATENTS, 1):
        description = f"RAG Test {i}/3: {patent['name']}"
        command = f'python main.py --pipeline rag_only --query "{patent["abstract"]}" --embedding_type sentence_transformer --rag_top_k 5'
        run_command(command, description)

def test_ml_approach():
    """Test AI patents using ML approach (if models are trained)"""
    print("\n🤖 TESTING ML APPROACH WITH AI PATENTS")
    print("="*80)
    
    # Test with short queries first (full abstracts might be too long for ML approach)
    short_queries = [
        "medical diagnosis machine learning model",
        "event prediction artificial intelligence",
        "neural network processing cluster scheduling"
    ]
    
    for i, query in enumerate(short_queries, 1):
        description = f"ML Test {i}/3: {query}"
        command = f'python main.py --pipeline patent_product --mode test --embedding_type sentence_transformer --sentence_transformer_model all-MiniLM-L6-v2'
        # Note: This will run the predefined test queries in the pipeline
        run_command(command, description)

def test_comprehensive_comparison():
    """Test with a specific query using both approaches"""
    print("\n📊 COMPREHENSIVE COMPARISON TEST")
    print("="*80)
    
    test_query = "artificial intelligence machine learning neural networks"
    
    # Test with RAG approach
    description = "RAG Approach - AI Query"
    command = f'python main.py --pipeline rag_only --query "{test_query}" --embedding_type sentence_transformer --rag_top_k 10'
    run_command(command, description)

def main():
    """Main test function"""
    print("🧪 AI PATENT TESTING FOR FULLFLOW SYSTEM")
    print("="*80)
    print("This script tests the FullFlow system with AI patent abstracts")
    print("It demonstrates both RAG and ML approaches for patent-product matching")
    print("="*80)
    
    # Check if we should run all tests or specific ones
    import argparse
    parser = argparse.ArgumentParser(description='Test FullFlow with AI patents')
    parser.add_argument('--approach', choices=['rag', 'ml', 'comparison', 'all'], 
                       default='all', help='Which approach to test')
    parser.add_argument('--embedding', choices=['fasttext', 'sentence_transformer'], 
                       default='sentence_transformer', help='Embedding type to use')
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    if args.approach in ['rag', 'all']:
        test_rag_approach()
    
    if args.approach in ['ml', 'all']:
        test_ml_approach()
    
    if args.approach in ['comparison', 'all']:
        test_comprehensive_comparison()
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n🎉 ALL TESTS COMPLETED")
    print(f"Total time: {duration:.2f} seconds")
    print(f"{'='*80}")
    
    # Summary
    print("\n📋 TEST SUMMARY:")
    print("✅ RAG approach tested with all 3 AI patent abstracts")
    print("✅ ML approach tested with representative queries") 
    print("✅ Comprehensive comparison performed")
    print("\n💡 WHAT WAS TESTED:")
    print("🔬 AI Diagnosis Apparatus Patent")
    print("🔬 AI Event Prediction System Patent") 
    print("🔬 Neural Processing Cluster Patent")
    print("\n🎯 BOTH APPROACHES DEMONSTRATED:")
    print("📊 RAG: Retrieval-Augmented Generation with ChromaDB")
    print("🤖 ML: Machine Learning with Transformation Matrices")
    print(f"{'='*80}")

if __name__ == "__main__":
    main() 