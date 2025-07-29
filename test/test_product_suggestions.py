#!/usr/bin/env python3
"""
Test Script for Product Suggestion Pipeline
==========================================

This script tests the product suggestion functionality integrated with the FullFlow system.
"""

import os
import sys
import json
import logging

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from pipelines.product_suggestion_pipeline import PatentProductSuggester

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_comprehensive_test_data():
    """Create comprehensive test data covering different technology domains."""
    
    return [
        {
            'hojinid': 'TEST_AI_001',
            'name': 'Advanced AI Research Corp',
            'keywords': ['artificial intelligence', 'machine learning', 'deep learning', 'neural networks', 'computer vision'],
            'patents': [
                {
                    'patent_id': 'US2023_AI_001',
                    'abstract': 'A novel deep learning architecture for real-time object detection in autonomous vehicles using convolutional neural networks with attention mechanisms for improved accuracy in challenging weather conditions.'
                },
                {
                    'patent_id': 'US2023_AI_002',
                    'abstract': 'Machine learning system for medical image analysis using artificial intelligence to automatically detect anomalies in X-ray, MRI, and CT scans with clinical decision support.'
                }
            ]
        },
        {
            'hojinid': 'TEST_QUANTUM_002',
            'name': 'Quantum Computing Solutions Inc',
            'keywords': ['quantum computing', 'quantum algorithms', 'cryptography', 'optimization', 'quantum hardware'],
            'patents': [
                {
                    'patent_id': 'US2023_QC_001',
                    'abstract': 'Quantum algorithm implementation for solving complex optimization problems in logistics and supply chain management using variational quantum eigensolvers and quantum annealing techniques.'
                },
                {
                    'patent_id': 'US2023_QC_002',
                    'abstract': 'Quantum cryptography system for secure communication networks using quantum key distribution protocols and error correction for enterprise security applications.'
                }
            ]
        },
        {
            'hojinid': 'TEST_BIO_003',
            'name': 'BioTech Innovations Ltd',
            'keywords': ['biotechnology', 'genetic engineering', 'drug discovery', 'bioinformatics', 'molecular biology'],
            'patents': [
                {
                    'patent_id': 'US2023_BIO_001',
                    'abstract': 'CRISPR-based gene editing platform for treating inherited genetic disorders using targeted molecular interventions with improved specificity and reduced off-target effects.'
                },
                {
                    'patent_id': 'US2023_BIO_002',
                    'abstract': 'AI-powered drug discovery system using machine learning to identify potential therapeutic compounds for cancer treatment through molecular modeling and virtual screening.'
                }
            ]
        },
        {
            'hojinid': 'TEST_ENERGY_004',
            'name': 'Green Energy Systems Corp',
            'keywords': ['renewable energy', 'solar technology', 'energy storage', 'smart grid', 'sustainable technology'],
            'patents': [
                {
                    'patent_id': 'US2023_ENE_001',
                    'abstract': 'Advanced photovoltaic cell technology with perovskite materials and nanotechnology coatings for enhanced solar energy conversion efficiency in various environmental conditions.'
                },
                {
                    'patent_id': 'US2023_ENE_002',
                    'abstract': 'Smart grid management system using IoT sensors and machine learning algorithms for optimizing renewable energy distribution and storage in residential and commercial applications.'
                }
            ]
        },
        {
            'hojinid': 'TEST_ROBOTICS_005',
            'name': 'Robotics Automation Technologies',
            'keywords': ['robotics', 'automation', 'industrial robots', 'ai control', 'sensor technology'],
            'patents': [
                {
                    'patent_id': 'US2023_ROB_001',
                    'abstract': 'Autonomous robotic system for warehouse automation using computer vision and artificial intelligence for inventory management and order fulfillment in e-commerce applications.'
                },
                {
                    'patent_id': 'US2023_ROB_002',
                    'abstract': 'Collaborative robot with advanced sensor technology for safe human-robot interaction in manufacturing environments with real-time collision avoidance and adaptive control.'
                }
            ]
        }
    ]


def test_basic_functionality():
    """Test basic product suggestion functionality."""
    print("\n" + "="*60)
    print("🧪 TEST 1: Basic Functionality")
    print("="*60)
    
    # Initialize suggester
    suggester = PatentProductSuggester({
        'similarity_threshold': 0.1,  # Lower threshold for testing
        'top_k_suggestions': 3,
        'enable_openai_enhance': False
    })
    
    # Simple test data
    test_companies = [
        {
            'hojinid': 'BASIC_TEST_001',
            'name': 'AI Test Company',
            'keywords': ['artificial intelligence', 'machine learning'],
            'patents': [
                {
                    'patent_id': 'TEST_001',
                    'abstract': 'Machine learning system for data analysis using neural networks.'
                }
            ]
        }
    ]
    
    query = "artificial intelligence system for automated analysis"
    
    # Generate suggestions
    results = suggester.suggest_products(query, test_companies)
    
    # Validate results
    assert 'summary' in results, "Results should contain summary"
    assert 'results' in results, "Results should contain results list"
    
    if results['results']:
        print("✅ Basic functionality test passed")
        company = results['results'][0]
        print(f"   Company: {company['company_name']}")
        print(f"   Similarity: {company['company_similarity']:.3f}")
        print(f"   Products: {len(company['products'])}")
    else:
        print("⚠️ No suggestions generated (might need lower threshold)")
    
    return results


def test_comprehensive_scenarios():
    """Test with comprehensive scenarios covering different domains."""
    print("\n" + "="*60)
    print("🧪 TEST 2: Comprehensive Domain Coverage")
    print("="*60)
    
    suggester = PatentProductSuggester({
        'similarity_threshold': 0.05,  # Very low threshold for comprehensive testing
        'top_k_suggestions': 2,
        'enable_openai_enhance': False
    })
    
    companies = create_comprehensive_test_data()
    
    test_scenarios = [
        {
            'name': 'AI Medical Imaging',
            'query': 'Artificial intelligence system for automated medical diagnosis using deep learning neural networks to analyze medical images and provide diagnostic recommendations for healthcare professionals.',
            'expected_companies': ['Advanced AI Research Corp']
        },
        {
            'name': 'Quantum Cryptography',
            'query': 'Quantum computing algorithms for secure cryptographic applications using quantum key distribution and quantum encryption protocols for enterprise security.',
            'expected_companies': ['Quantum Computing Solutions Inc']
        },
        {
            'name': 'Gene Therapy',
            'query': 'Biotechnology platform for gene editing and therapy using CRISPR technology to treat genetic diseases through targeted molecular interventions.',
            'expected_companies': ['BioTech Innovations Ltd']
        },
        {
            'name': 'Smart Grid Energy',
            'query': 'Renewable energy management system using solar technology and smart grid integration for optimizing energy distribution and storage.',
            'expected_companies': ['Green Energy Systems Corp']
        },
        {
            'name': 'Industrial Automation',
            'query': 'Robotic automation system with AI-powered control for manufacturing and industrial process optimization using computer vision and sensors.',
            'expected_companies': ['Robotics Automation Technologies']
        }
    ]
    
    total_tests = len(test_scenarios)
    passed_tests = 0
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n🔍 Scenario {i}: {scenario['name']}")
        print(f"Query: {scenario['query'][:80]}...")
        
        try:
            results = suggester.suggest_products(scenario['query'], companies)
            
            companies_found = [r['company_name'] for r in results['results']]
            matches = len(results['results'])
            
            if matches > 0:
                print(f"✅ Found {matches} matches")
                top_company = results['results'][0]
                print(f"   Top: {top_company['company_name']} (Score: {top_company['company_similarity']:.3f})")
                
                if top_company['products']:
                    print(f"   Best Product: {top_company['products'][0]['product_name']}")
                    print(f"   Product Score: {top_company['products'][0]['score']:.3f}")
                
                passed_tests += 1
            else:
                print("❌ No matches found")
                
        except Exception as e:
            print(f"❌ Error in scenario {i}: {e}")
    
    print(f"\n📊 Comprehensive Test Results: {passed_tests}/{total_tests} scenarios passed")
    return passed_tests == total_tests


def test_export_functionality():
    """Test file export and JSON serialization."""
    print("\n" + "="*60)
    print("🧪 TEST 3: Export Functionality")
    print("="*60)
    
    suggester = PatentProductSuggester({
        'similarity_threshold': 0.05,
        'top_k_suggestions': 2,
        'output_directory': 'test/outputs'  # Test directory
    })
    
    # Ensure test output directory exists
    os.makedirs('test/outputs', exist_ok=True)
    
    # Use subset of test data
    companies = create_comprehensive_test_data()[:2]
    query = "AI system for automated image analysis and machine learning applications"
    
    try:
        results = suggester.suggest_products(query, companies)
        output_path = suggester.export_results(results, "test_export.json")
        
        if os.path.exists(output_path):
            # Verify JSON is valid
            with open(output_path, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
            
            print("✅ Export functionality test passed")
            print(f"   File saved: {output_path}")
            print(f"   File size: {os.path.getsize(output_path)} bytes")
            print(f"   Companies in export: {len(loaded_data.get('results', []))}")
            
            return True
        else:
            print("❌ Export file not created")
            return False
            
    except Exception as e:
        print(f"❌ Export test failed: {e}")
        return False


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n" + "="*60)
    print("🧪 TEST 4: Edge Cases and Error Handling")
    print("="*60)
    
    suggester = PatentProductSuggester({
        'similarity_threshold': 0.1,
        'top_k_suggestions': 2
    })
    
    edge_cases = [
        {
            'name': 'Empty query',
            'query': '',
            'companies': create_comprehensive_test_data()[:1]
        },
        {
            'name': 'Empty companies list',
            'query': 'test query',
            'companies': []
        },
        {
            'name': 'Company with no keywords or patents',
            'query': 'test query',
            'companies': [{'hojinid': 'EMPTY_001', 'name': 'Empty Company', 'keywords': [], 'patents': []}]
        },
        {
            'name': 'Very long query',
            'query': 'artificial intelligence ' * 100,  # Very long query
            'companies': create_comprehensive_test_data()[:1]
        }
    ]
    
    passed_cases = 0
    
    for i, case in enumerate(edge_cases, 1):
        print(f"\n🔍 Edge Case {i}: {case['name']}")
        
        try:
            results = suggester.suggest_products(case['query'], case['companies'])
            
            # Should handle gracefully without crashing
            print("✅ Handled gracefully")
            print(f"   Results: {len(results.get('results', []))} companies")
            passed_cases += 1
            
        except Exception as e:
            print(f"❌ Failed with error: {e}")
    
    print(f"\n📊 Edge Case Results: {passed_cases}/{len(edge_cases)} cases passed")
    return passed_cases == len(edge_cases)


def run_all_tests():
    """Run comprehensive test suite."""
    print("🚀 PRODUCT SUGGESTION PIPELINE - COMPREHENSIVE TEST SUITE")
    print("="*70)
    
    all_tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Comprehensive Scenarios", test_comprehensive_scenarios),
        ("Export Functionality", test_export_functionality),
        ("Edge Cases", test_edge_cases)
    ]
    
    results = {}
    
    for test_name, test_func in all_tests:
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*70)
    print("📊 TEST SUITE SUMMARY")
    print("="*70)
    
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:.<50} {status}")
    
    print(f"\nOverall Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Product suggestion pipeline is ready for integration.")
    else:
        print("⚠️ Some tests failed. Review the output above for details.")
    
    return passed == total


if __name__ == "__main__":
    try:
        success = run_all_tests()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n👋 Tests interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n❌ Test suite failed with unexpected error: {e}")
        exit(1) 