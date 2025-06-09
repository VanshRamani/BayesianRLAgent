#!/usr/bin/env python3
"""
Quick test to verify ArXiv API works
"""

import sys
sys.path.append('src')

from discovery.paper_finder import PaperFinder

def test_arxiv():
    print("🧪 Testing ArXiv API...")
    
    try:
        pf = PaperFinder()
        papers = pf.find_arxiv_papers(days_back=30)
        
        print(f"✅ Found {len(papers)} papers:")
        for i, paper in enumerate(papers[:3], 1):  # Show first 3
            print(f"   {i}. {paper.title}")
            print(f"      Authors: {', '.join(paper.authors[:2])}")
            print(f"      Published: {paper.published_date}")
            print(f"      Categories: {', '.join(paper.categories)}")
            print()
        
        return True
        
    except Exception as e:
        print(f"❌ ArXiv test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_arxiv() 