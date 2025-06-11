#!/usr/bin/env python3
"""
Real demo for Bayesian RL Agent - uses actual ArXiv papers with LLM analysis
"""

import sys
import os
from datetime import datetime, timedelta

# Add src to path
sys.path.append('src')

from analysis.belief_tracker import BeliefTracker
from analysis.llm_analyzer import LLMAnalyzer
from discovery.paper_finder import PaperFinder
import json


def run_real_demo():
    """Run demo with real ArXiv papers and LLM analysis"""
    print("🚀 Bayesian RL Agent - Real Paper Demo")
    print("=" * 60)
    print("Fetching real papers from ArXiv and analyzing with LLM...")
    
    # Check for OpenAI API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ Error: OpenAI API key required!")
        print("Set environment variable: export OPENAI_API_KEY='your-key-here'")
        print("\n💡 Alternative: You can run without LLM using:")
        print("   python simple_demo.py  # Uses mock data")
        return
    
    # Initialize components
    print("\n1. Initializing components...")
    tracker = BeliefTracker()
    analyzer = LLMAnalyzer(api_key=api_key)
    paper_finder = PaperFinder()
    
    # Find recent RL papers
    print("\n2. Searching ArXiv for recent RL papers...")
    papers = paper_finder.find_arxiv_papers(days_back=30)
    
    if not papers:
        print("❌ No papers found. Check your internet connection or try a wider date range.")
        return
    
    # Limit to first 3 papers for testing
    papers = papers[:3]
    
    print(f"✅ Found {len(papers)} papers (showing first 3 for demo):")
    for i, paper in enumerate(papers, 1):
        print(f"   {i}. {paper.title[:60]}...")
        print(f"      Authors: {', '.join(paper.authors[:2])}{'...' if len(paper.authors) > 2 else ''}")
        print(f"      Published: {paper.published_date.strftime('%Y-%m-%d')}")
    
    # Analyze papers with LLM
    print(f"\n3. Analyzing papers with LLM (this may take a minute)...")
    print(f"   Using model: {analyzer.model}")
    all_evidence = []
    
    for i, paper in enumerate(papers, 1):
        print(f"   Analyzing paper {i}/{len(papers)}: {paper.title[:40]}...")
        
        try:
            evidence_list = analyzer.analyze_paper(paper)
            all_evidence.extend(evidence_list)
            
            if evidence_list:
                print(f"      ✅ Found {len(evidence_list)} technique mentions")
                for evidence in evidence_list:
                    print(f"         - {evidence.technique}: {evidence.value:.3f} (conf: {evidence.confidence:.3f})")
            else:
                print(f"      ⚪ No specific RL techniques detected")
                
        except Exception as e:
            print(f"      ❌ Error: {e}")
    
    if not all_evidence:
        print("\n❌ No evidence extracted from papers.")
        print("💡 This could mean:")
        print("   - Papers don't mention specific RL techniques")
        print("   - LLM API issues")
        print("   - Need to adjust technique detection")
        return
    
    # Update beliefs with real evidence
    print(f"\n4. Updating Bayesian beliefs...")
    print(f"   Processing {len(all_evidence)} pieces of evidence...")
    
    for evidence in all_evidence:
        tracker.update_belief(evidence)
        print(f"   Updated {evidence.technique}: effectiveness={evidence.value:.3f}, confidence={evidence.confidence:.3f}")
    
    # Show results
    print(f"\n5. Results after analyzing real papers:")
    summary = tracker.generate_summary()
    
    print(f"\n📊 Overview:")
    print(f"   Total techniques tracked: {summary['total_techniques']}")
    print(f"   Total evidence points: {summary['total_evidence']}")
    print(f"   Papers analyzed: {len(papers)}")
    
    if summary.get("most_promising"):
        print(f"\n🚀 Most promising techniques (from real papers):")
        for i, tech in enumerate(summary["most_promising"][:5], 1):
            belief = tracker.beliefs[tech]
            ci_low, ci_high = belief.confidence_interval(0.95)
            print(f"   {i}. {tech:<15} effectiveness: {belief.mean_effectiveness:.3f} "
                  f"[{ci_low:.3f}-{ci_high:.3f}] (certainty: {belief.certainty:.2f})")
    
    # Show paper sources for each technique
    print(f"\n📄 Evidence sources by technique:")
    for technique in tracker.beliefs.keys():
        technique_evidence = [e for e in all_evidence if e.technique == technique]
        if technique_evidence:
            print(f"\n   {technique}:")
            for evidence in technique_evidence:
                reasoning = evidence.context.get('llm_reasoning', 'No reasoning provided')[:100]
                print(f"     • {evidence.source}")
                print(f"       Score: {evidence.value:.3f}, LLM reasoning: {reasoning}...")
    
    # Save everything
    beliefs_file = tracker.save_beliefs()
    
    # Save paper data
    papers_data = []
    for paper in papers:
        papers_data.append({
            "title": paper.title,
            "authors": paper.authors,
            "abstract": paper.abstract,
            "url": paper.url,
            "published_date": paper.published_date.isoformat(),
            "categories": paper.categories
        })
    
    papers_file = f"data/papers/papers_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs(os.path.dirname(papers_file), exist_ok=True)
    with open(papers_file, 'w') as f:
        json.dump(papers_data, f, indent=2)
    
    print(f"\n✅ Real paper analysis completed!")
    print(f"   🧠 Beliefs saved to: {beliefs_file}")
    print(f"   📄 Paper data saved to: {papers_file}")
    
    print(f"\n💡 What happened:")
    print(f"   - Fetched {len(papers)} real papers from ArXiv")
    print(f"   - Used GPT-4o-mini to analyze technique effectiveness")
    print(f"   - Extracted {len(all_evidence)} evidence points")
    print(f"   - Updated Bayesian beliefs based on LLM assessments")
    print(f"   - No more fake data - this is based on real research!")
    
    print(f"\n🚀 Next steps:")
    print(f"   - Set up daily automation: python -m src.agent.daily_run")
    print(f"   - Add more papers: modify days_back parameter")
    print(f"   - View beliefs: python -m src.analysis.view_beliefs")


if __name__ == "__main__":
    run_real_demo() 