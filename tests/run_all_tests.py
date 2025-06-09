#!/usr/bin/env python3
"""
Run all tests to verify the complete Bayesian RL Agent system
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def run_all_tests():
    """Run complete test suite"""
    print("🧪 Bayesian RL Agent - Complete Test Suite")
    print("=" * 60)
    
    tests = [
        ("ArXiv API", "tests/test_arxiv.py"),
        ("Gemini API", "tests/test_gemini.py"), 
        ("Bayesian Beliefs", "tests/test_beliefs.py"),
        ("Full Pipeline", "tests/test_full_pipeline.py")
    ]
    
    results = {}
    
    for test_name, test_file in tests:
        print(f"\n🚀 Running {test_name} test...")
        print("-" * 40)
        
        try:
            # Import and run each test
            if test_name == "ArXiv API":
                from test_arxiv import test_arxiv_basic
                success, data = test_arxiv_basic()
                results[test_name] = {"success": success, "data": f"{len(data)} papers" if success else "Failed"}
            
            elif test_name == "Gemini API":
                from test_gemini import test_gemini_basic
                success, data = test_gemini_basic()
                results[test_name] = {"success": success, "data": f"{len(data.get('techniques', []))} techniques" if success else "Failed"}
            
            elif test_name == "Bayesian Beliefs":
                from test_beliefs import test_belief_tracking
                success = test_belief_tracking()
                results[test_name] = {"success": success, "data": "Belief tracking working" if success else "Failed"}
            
            elif test_name == "Full Pipeline":
                from test_full_pipeline import test_full_pipeline
                success = test_full_pipeline()
                results[test_name] = {"success": success, "data": "End-to-end working" if success else "Failed"}
                
        except Exception as e:
            print(f"❌ {test_name} test failed with error: {e}")
            results[test_name] = {"success": False, "data": str(e)}
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result["success"] else "❌ FAIL"
        print(f"{status} {test_name:<20} - {result['data']}")
        if result["success"]:
            passed += 1
    
    print("-" * 60)
    print(f"📈 OVERALL: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! 🚀")
        print("✨ Your Bayesian RL Agent is ready for real research analysis!")
        print("\n🔥 Key Features Working:")
        print("   • Real ArXiv paper discovery")
        print("   • Gemini AI analysis of research content")
        print("   • Bayesian belief updating with uncertainty")
        print("   • End-to-end pipeline integration")
        print("\n💡 Next steps:")
        print("   • Run: python tests/test_full_pipeline.py")
        print("   • Set up daily automation")
        print("   • Analyze more papers for better beliefs")
    else:
        print(f"\n⚠️  {total-passed} tests failed. Check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 