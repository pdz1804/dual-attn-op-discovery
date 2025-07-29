#!/usr/bin/env python3
"""
Simple Test Script for PatentProductSuggester
============================================

This script demonstrates the core functionality of the patent-based product suggester
system with a focused test case.
"""

from product_suggester import PatentProductSuggester, create_test_data
import json

def simple_test():
    """Run a simple, focused test of the product suggester."""
    
    print("🚀 PATENT PRODUCT SUGGESTER - SIMPLE TEST")
    print("=" * 50)
    
    # Initialize with reasonable parameters
    suggester = PatentProductSuggester(
        alpha=0.6,      # Balance keyword vs patent similarity
        beta=0.5,       # Balance lexical vs semantic similarity
        similarity_threshold=0.15,  # Reasonable threshold
        top_k_suggestions=2,        # Limit suggestions for clarity
        use_patent_data=True
    )
    
    # Create focused test data
    companies = [
        {
            'hojinid': 'TEST001',
            'name': 'MedTech AI Solutions',
            'keywords': ['artificial intelligence', 'medical imaging', 'deep learning', 'computer vision', 'healthcare'],
            'patents': [
                {
                    'patent_id': 'US2023001',
                    'abstract': 'Deep learning system for medical image analysis using convolutional neural networks to detect tumors in MRI and CT scans with high accuracy.'
                },
                {
                    'patent_id': 'US2023002',
                    'abstract': 'AI-powered diagnostic tool for radiology using machine learning algorithms to assist healthcare professionals in medical imaging analysis.'
                }
            ]
        },
        {
            'hojinid': 'TEST002',
            'name': 'Vision Robotics Corp',
            'keywords': ['computer vision', 'robotics', 'automation', 'machine learning', 'sensors'],
            'patents': [
                {
                    'patent_id': 'US2023003',
                    'abstract': 'Robotic vision system using advanced computer vision algorithms for industrial automation and quality control applications.'
                }
            ]
        }
    ]
    
    # Test query that should match both companies
    query = """
    Advanced artificial intelligence system for medical image processing using 
    deep learning neural networks to automatically detect and classify medical 
    conditions in radiological images, providing diagnostic support for healthcare professionals.
    """
    
    print(f"📝 Query: {query.strip()[:100]}...")
    print()
    
    # Get suggestions
    print("🔍 Processing suggestions...")
    results = suggester.suggest_products(query, companies)
    
    # Display detailed results
    print("\n📊 RESULTS:")
    print(f"Companies processed: {results['summary']['total_companies_processed']}")
    print(f"Companies with suggestions: {results['summary']['companies_with_suggestions']}")
    print(f"Total products suggested: {results['summary']['total_products_suggested']}")
    
    if results['results']:
        for i, company in enumerate(results['results'], 1):
            print(f"\n🏢 {i}. {company['company_name']} (ID: {company['hojinid']})")
            print(f"   📈 Company Similarity: {company['company_similarity']:.3f}")
            print(f"   📋 Keyword Similarity: {company['similarity_breakdown']['keyword_similarity']:.3f}")
            print(f"   📄 Patent Similarity: {company['similarity_breakdown']['patent_similarity']:.3f}")
            print(f"   🧪 Patents Analyzed: {company['patent_analysis']['patent_count']}")
            print(f"   🔑 Key Themes: {', '.join(company['patent_analysis']['themes'][:3])}")
            
            print("   🎯 Suggested Products:")
            for j, product in enumerate(company['products'], 1):
                print(f"      {j}. {product['product_name']}")
                print(f"         Score: {product['score']:.3f} | "
                      f"Lexical: {product['similarity_breakdown']['lexical']:.3f} | "
                      f"Semantic: {product['similarity_breakdown']['semantic']:.3f}")
    else:
        print("❌ No suggestions found. Try lowering the similarity threshold.")
    
    # Export results
    output_file = "simple_test_results.json"
    if suggester.export_results(results, output_file):
        print(f"\n💾 Results saved to: {output_file}")
    
    return results

def performance_test():
    """Test with multiple queries to show different matching scenarios."""
    
    print("\n" + "=" * 50)
    print("🚀 PERFORMANCE TEST - Multiple Queries")
    print("=" * 50)
    
    suggester = PatentProductSuggester(
        similarity_threshold=0.1,  # Lower threshold for demonstration
        top_k_suggestions=2
    )
    
    # Use the full test dataset
    companies = create_test_data()
    
    test_scenarios = [
        {
            'name': 'AI Medical Imaging',
            'query': 'Artificial intelligence for medical image analysis and diagnostic support'
        },
        {
            'name': 'Quantum Computing',
            'query': 'Quantum algorithms for cryptographic security and optimization problems'
        },
        {
            'name': 'Renewable Energy',
            'query': 'Solar energy systems with smart grid integration and storage solutions'
        },
        {
            'name': 'Robotics Automation',
            'query': 'Industrial robotics with computer vision for manufacturing automation'
        }
    ]
    
    for scenario in test_scenarios:
        print(f"\n🔍 Scenario: {scenario['name']}")
        print(f"Query: {scenario['query']}")
        
        results = suggester.suggest_products(scenario['query'], companies)
        matches = len(results['results'])
        
        if matches > 0:
            top_company = results['results'][0]
            print(f"✅ Found {matches} matches | "
                  f"Top: {top_company['company_name']} "
                  f"(Score: {top_company['company_similarity']:.3f})")
            
            if top_company['products']:
                print(f"   Best Product: {top_company['products'][0]['product_name']}")
        else:
            print("❌ No matches found")
    
    print(f"\n✨ Performance test completed!")

if __name__ == "__main__":
    print("Select test:")
    print("1. Simple focused test")
    print("2. Performance test with multiple scenarios")
    print("3. Both")
    
    try:
        choice = input("Enter choice (1-3): ").strip()
        
        if choice == "1":
            simple_test()
        elif choice == "2":
            performance_test()
        elif choice == "3":
            simple_test()
            performance_test()
        else:
            print("Running simple test by default...")
            simple_test()
            
    except KeyboardInterrupt:
        print("\n\n👋 Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during testing: {e}") 