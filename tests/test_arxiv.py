#!/usr/bin/env python3
"""
Test ArXiv API functionality
"""

import sys
import os
from dotenv import load_dotenv
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from discovery.paper_finder import PaperFinder

load_dotenv()

def test_arxiv_basic():
    """Test basic ArXiv functionality"""
    print("🧪 Testing ArXiv API...")
    
    try:
        pf = PaperFinder()
        papers = pf.find_arxiv_papers(days_back=30)
        
        print(f"✅ Found {len(papers)} papers")
        if papers:
            print("\n📄 Sample papers:")
            for i, paper in enumerate(papers[:3], 1):
                print(f"   {i}. {paper.title}")
                print(f"      Authors: {', '.join(paper.authors[:2])}")
                print(f"      Published: {paper.published_date}")
                print(f"      Categories: {', '.join(paper.categories)}")
                print(f"      Abstract length: {len(paper.abstract)} chars")
                print()
        
        return True, papers
        
    except Exception as e:
        print(f"❌ ArXiv test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, []

if __name__ == "__main__":
    success, papers = test_arxiv_basic()
    if success:
        print(f"🎉 ArXiv test passed! Found {len(papers)} papers")
    else:
        print("💥 ArXiv test failed") 