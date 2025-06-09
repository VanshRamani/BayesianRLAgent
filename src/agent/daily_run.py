"""
Main daily agent that coordinates discovery, analysis, and reporting
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from discovery.paper_finder import PaperFinder
from discovery.repo_finder import RepoFinder
from analysis.belief_tracker import BeliefTracker, Evidence, EvidenceType
from analysis.llm_analyzer import LLMAnalyzer
from reporting.daily_report import DailyReporter

from datetime import datetime
import json
from typing import List, Dict


class BayesianRLAgent:
    """Main agent that orchestrates daily discovery and belief updates"""
    
    def __init__(self, github_token: str = None, openai_api_key: str = None):
        print("🤖 Initializing Bayesian RL Agent...")
        
        self.paper_finder = PaperFinder()
        self.repo_finder = RepoFinder(github_token=github_token)
        self.belief_tracker = BeliefTracker()
        
        # Use LLM analyzer instead of regex-based analyzer
        try:
            self.content_analyzer = LLMAnalyzer(api_key=openai_api_key)
            print("✅ LLM analyzer initialized with OpenAI API")
        except ValueError as e:
            print(f"❌ LLM analyzer initialization failed: {e}")
            print("💡 Falling back to regex-based analyzer (less accurate)")
            from analysis.content_analyzer import ContentAnalyzer
            self.content_analyzer = ContentAnalyzer()
        
        self.reporter = DailyReporter()
        
        print("✅ Agent initialized successfully!")
    
    def run_daily_cycle(self) -> Dict:
        """Execute complete daily discovery and belief update cycle"""
        print(f"\n🚀 Starting daily cycle at {datetime.now()}")
        
        cycle_results = {
            "timestamp": datetime.now().isoformat(),
            "papers_found": 0,
            "repos_found": 0,
            "beliefs_updated": 0,
            "new_techniques": 0,
            "report_generated": False,
            "analyzer_type": "LLM" if isinstance(self.content_analyzer, LLMAnalyzer) else "Regex"
        }
        
        try:
            # 1. Discover new papers
            print("\n📰 Discovering new papers...")
            papers = self.paper_finder.discover_daily()
            cycle_results["papers_found"] = len(papers)
            
            if papers:
                papers_file = self.paper_finder.save_papers(papers)
                print(f"💾 Saved {len(papers)} papers to {papers_file}")
            
            # 2. Discover new repositories  
            print("\n🔍 Discovering new repositories...")
            repos = self.repo_finder.discover_daily()
            cycle_results["repos_found"] = len(repos)
            
            if repos:
                repos_file = self.repo_finder.save_repos(repos)
                print(f"💾 Saved {len(repos)} repos to {repos_file}")
            
            # 3. Analyze content and extract evidence
            print(f"\n🧠 Analyzing content with {cycle_results['analyzer_type']} analyzer...")
            evidence_list = []
            
            # Analyze papers for technique mentions and effectiveness indicators
            for i, paper in enumerate(papers, 1):
                print(f"   Analyzing paper {i}/{len(papers)}: {paper.title[:50]}...")
                try:
                    paper_evidence = self.content_analyzer.analyze_paper(paper)
                    evidence_list.extend(paper_evidence)
                    if paper_evidence:
                        print(f"      ✅ Found {len(paper_evidence)} technique mentions")
                    else:
                        print(f"      ⚪ No techniques detected")
                except Exception as e:
                    print(f"      ❌ Error analyzing paper: {e}")
            
            # Analyze repos for technique adoption and popularity
            for i, repo in enumerate(repos, 1):
                print(f"   Analyzing repo {i}/{len(repos)}: {repo.name}...")
                try:
                    repo_evidence = self.content_analyzer.analyze_repo(repo)
                    evidence_list.extend(repo_evidence)
                    if repo_evidence:
                        print(f"      ✅ Found {len(repo_evidence)} technique mentions")
                    else:
                        print(f"      ⚪ No techniques detected")
                except Exception as e:
                    print(f"      ❌ Error analyzing repo: {e}")
            
            print(f"📊 Extracted {len(evidence_list)} pieces of evidence total")
            
            # 4. Update beliefs
            print("\n🎯 Updating beliefs...")
            techniques_before = len(self.belief_tracker.beliefs)
            
            for evidence in evidence_list:
                self.belief_tracker.update_belief(evidence)
                print(f"   Updated {evidence.technique}: {evidence.value:.3f} (conf: {evidence.confidence:.3f})")
            
            techniques_after = len(self.belief_tracker.beliefs)
            cycle_results["beliefs_updated"] = len(evidence_list)
            cycle_results["new_techniques"] = techniques_after - techniques_before
            
            # Save updated beliefs
            beliefs_file = self.belief_tracker.save_beliefs()
            print(f"💾 Saved updated beliefs to {beliefs_file}")
            
            # 5. Generate daily report
            print("\n📝 Generating daily report...")
            report_data = {
                "papers": papers,
                "repos": repos,
                "evidence": evidence_list,
                "belief_summary": self.belief_tracker.generate_summary(),
                "cycle_results": cycle_results
            }
            
            report_file = self.reporter.generate_report(report_data)
            cycle_results["report_generated"] = True
            print(f"📄 Generated report: {report_file}")
            
            # 6. Print summary
            self._print_cycle_summary(cycle_results)
            
        except Exception as e:
            print(f"❌ Error during daily cycle: {e}")
            cycle_results["error"] = str(e)
        
        return cycle_results
    
    def _print_cycle_summary(self, results: Dict):
        """Print a nice summary of the daily cycle"""
        print("\n" + "="*50)
        print("📈 DAILY CYCLE SUMMARY")
        print("="*50)
        print(f"📰 Papers discovered: {results['papers_found']}")
        print(f"🔍 Repositories found: {results['repos_found']}")
        print(f"🎯 Beliefs updated: {results['beliefs_updated']}")
        print(f"✨ New techniques tracked: {results['new_techniques']}")
        print(f"🤖 Analyzer used: {results['analyzer_type']}")
        print(f"📄 Report generated: {'✅' if results['report_generated'] else '❌'}")
        
        # Show current top beliefs
        summary = self.belief_tracker.generate_summary()
        if "most_promising" in summary and summary["most_promising"]:
            print(f"\n🚀 Most promising techniques:")
            for i, tech in enumerate(summary["most_promising"][:5], 1):
                belief = self.belief_tracker.beliefs[tech]
                print(f"   {i}. {tech} (effectiveness: {belief.mean_effectiveness:.3f})")
        
        if "most_overhyped" in summary and summary["most_overhyped"]:
            print(f"\n😮‍💨 Most overhyped techniques:")
            for i, tech in enumerate(summary["most_overhyped"][:3], 1):
                belief = self.belief_tracker.beliefs[tech]
                print(f"   {i}. {tech} (effectiveness: {belief.mean_effectiveness:.3f})")
        
        print("\n✅ Daily cycle completed successfully!")
        print("="*50)


def main():
    """Main entry point for daily agent run"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Bayesian RL Agent daily cycle")
    parser.add_argument("--github-token", type=str, help="GitHub API token for higher rate limits")
    parser.add_argument("--openai-api-key", type=str, help="OpenAI API key for LLM analysis")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Get API keys from environment if not provided
    github_token = args.github_token or os.getenv('GITHUB_TOKEN')
    openai_key = args.openai_api_key or os.getenv('OPENAI_API_KEY')
    
    # Initialize and run agent
    agent = BayesianRLAgent(github_token=github_token, openai_api_key=openai_key)
    results = agent.run_daily_cycle()
    
    # Exit with appropriate code
    if "error" in results:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main() 