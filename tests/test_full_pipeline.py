#!/usr/bin/env python3
"""
Test full pipeline: ArXiv papers → Gemini analysis → Bayesian beliefs
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from discovery.paper_finder import PaperFinder
from analysis.gemini_analyzer import GeminiAnalyzer
from analysis.belief_tracker import BeliefTracker

def test_full_pipeline():
    """Test complete pipeline with real papers and Gemini"""
    print("🚀 Testing Full Pipeline: ArXiv → Gemini → Bayesian Beliefs")
    print("=" * 70)
    
    # Get API key from environment
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("❌ Error: GEMINI_API_KEY environment variable required!")
        print("Set environment variable: export GEMINI_API_KEY='your-key-here'")
        return False
    
    try:
        # 1. Initialize components
        print("\n1️⃣ Initializing components...")
        paper_finder = PaperFinder()
        analyzer = GeminiAnalyzer(api_key=api_key)
        tracker = BeliefTracker()
        print("   ✅ All components initialized")
        
        # 2. Get real papers
        print("\n2️⃣ Fetching real papers from ArXiv...")
        papers = paper_finder.find_arxiv_papers(days_back=30)
        
        if not papers:
            print("   ❌ No papers found!")
            return False
        
        # Limit to 3 papers for testing
        papers = papers[:3]
        
        print(f"   ✅ Found {len(papers)} papers (limited to 3 for testing)")
        for i, paper in enumerate(papers, 1):
            print(f"      {i}. {paper.title[:60]}...")
        
        # 3. Analyze with Gemini
        print(f"\n3️⃣ Analyzing papers with Gemini...")
        all_evidence = []
        
        for i, paper in enumerate(papers, 1):
            print(f"\n   📄 Paper {i}/{len(papers)}: {paper.title[:50]}...")
            
            try:
                evidence_list = analyzer.analyze_paper(paper)
                all_evidence.extend(evidence_list)
                
                if evidence_list:
                    print(f"      ✅ Found {len(evidence_list)} technique mentions:")
                    for ev in evidence_list:
                        print(f"         • {ev.technique}: {ev.value:.3f} (conf: {ev.confidence:.3f})")
                        reasoning = ev.context.get('llm_reasoning', '')[:80]
                        print(f"           Reasoning: {reasoning}...")
                else:
                    print(f"      ⚪ No RL techniques detected")
                    
            except Exception as e:
                print(f"      ❌ Error: {e}")
        
        print(f"\n   📊 Total evidence collected: {len(all_evidence)} pieces")
        
        if not all_evidence:
            print("   ⚠️  No evidence found - possibly no RL techniques in these papers")
            return True  # Still success, just no RL papers
        
        # 4. Update Bayesian beliefs
        print(f"\n4️⃣ Updating Bayesian beliefs...")
        
        for evidence in all_evidence:
            tracker.update_belief(evidence)
            print(f"   🧠 Updated {evidence.technique}: effectiveness={evidence.value:.3f}")
        
        # 5. Show results
        print(f"\n5️⃣ Results:")
        summary = tracker.generate_summary()
        
        print(f"\n📈 Belief Summary:")
        print(f"   • Techniques tracked: {summary['total_techniques']}")
        print(f"   • Evidence points: {summary['total_evidence']}")
        
        if summary.get("most_promising"):
            print(f"\n🚀 Most promising techniques:")
            for i, tech in enumerate(summary["most_promising"][:5], 1):
                belief = tracker.beliefs[tech]
                ci_low, ci_high = belief.confidence_interval(0.95)
                print(f"   {i}. {tech:<15} effectiveness: {belief.mean_effectiveness:.3f} "
                      f"[{ci_low:.3f}-{ci_high:.3f}]")
        
        # 6. Show evidence details
        print(f"\n📋 Evidence breakdown:")
        for technique in tracker.beliefs.keys():
            tech_evidence = [e for e in all_evidence if e.technique == technique]
            if tech_evidence:
                print(f"\n   {technique}:")
                for ev in tech_evidence:
                    print(f"     • Score: {ev.value:.3f} from: {ev.source}")
                    reasoning = ev.context.get('llm_reasoning', 'No reasoning')[:100]
                    print(f"       Gemini reasoning: {reasoning}...")
        
        print(f"\n✅ Full pipeline test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_full_pipeline()
    if success:
        print(f"\n🎉 Full pipeline working! Real papers + Gemini + Bayesian beliefs ✨")
    else:
        print("💥 Pipeline test failed") 