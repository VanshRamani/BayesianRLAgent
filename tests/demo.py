#!/usr/bin/env python3
"""
Demo script for Bayesian RL Agent - shows the system working with example data
"""

import os
import sys
from datetime import datetime, timedelta

# Add src to path
sys.path.append('src')

from discovery.paper_finder import Paper
from discovery.repo_finder import Repository
from analysis.belief_tracker import BeliefTracker, Evidence, EvidenceType
from analysis.content_analyzer import ContentAnalyzer
from reporting.daily_report import DailyReporter


def create_demo_papers():
    """Create some example papers for demonstration"""
    papers = [
        Paper(
            title="Proximal Policy Optimization Algorithms",
            authors=["John Schulman", "Filip Wolski", "Prafulla Dhariwal"],
            abstract="We propose a new family of policy gradient methods for reinforcement learning, which alternate between sampling data through interaction with the environment and optimizing a surrogate objective function using stochastic gradient ascent. PPO strikes a balance between ease of implementation, sample complexity, and ease of tuning, outperforming other methods.",
            url="https://arxiv.org/abs/1707.06347",
            published_date=datetime.now() - timedelta(days=1),
            categories=["cs.LG"],
            source="arxiv"
        ),
        Paper(
            title="Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning",
            authors=["Tuomas Haarnoja", "Aurick Zhou", "Pieter Abbeel"],
            abstract="Model-free deep reinforcement learning algorithms have been demonstrated on a range of challenging decision making and control tasks. However, these methods typically suffer from poor sample efficiency. SAC is an off-policy actor-critic deep RL algorithm based on the maximum entropy reinforcement learning framework, achieving state-of-the-art performance.",
            url="https://arxiv.org/abs/1801.01290",
            published_date=datetime.now() - timedelta(days=2),
            categories=["cs.LG", "cs.AI"],
            source="arxiv"
        ),
        Paper(
            title="Challenges in Deep Q-Network Implementation",
            authors=["Research Team"],
            abstract="This paper discusses various implementation challenges and failure modes observed in Deep Q-Networks. Despite their theoretical appeal, DQN methods often struggle with instability and poor convergence in practice. We analyze common pitfalls and provide recommendations.",
            url="https://example.com/dqn-challenges",
            published_date=datetime.now() - timedelta(days=3),
            categories=["cs.LG"],
            source="arxiv"
        )
    ]
    return papers


def create_demo_repos():
    """Create some example repositories for demonstration"""
    repos = [
        Repository(
            name="stable-baselines3",
            full_name="DLR-RM/stable-baselines3",
            description="PyTorch version of Stable Baselines, reliable implementations of reinforcement learning algorithms including PPO, SAC, TD3",
            url="https://github.com/DLR-RM/stable-baselines3",
            stars=4500,
            forks=850,
            language="Python",
            created_at=datetime.now() - timedelta(days=365),
            updated_at=datetime.now() - timedelta(days=1),
            topics=["reinforcement-learning", "machine-learning", "pytorch", "ppo", "sac"]
        ),
        Repository(
            name="rainbow",
            full_name="Kaixhin/Rainbow",
            description="Rainbow: Combining Improvements in Deep Reinforcement Learning",
            url="https://github.com/Kaixhin/Rainbow",
            stars=1200,
            forks=245,
            language="Python",
            created_at=datetime.now() - timedelta(days=800),
            updated_at=datetime.now() - timedelta(days=30),
            topics=["deep-reinforcement-learning", "dqn", "rainbow"]
        ),
        Repository(
            name="experimental-rl",
            full_name="researcher/experimental-rl",
            description="Experimental implementations of new RL algorithms",
            url="https://github.com/researcher/experimental-rl",
            stars=15,
            forks=3,
            language="Python",
            created_at=datetime.now() - timedelta(days=90),
            updated_at=datetime.now() - timedelta(days=5),
            topics=["reinforcement-learning", "research"]
        )
    ]
    return repos


def run_demo():
    """Run the complete demo"""
    print("🚀 Bayesian RL Agent Demo")
    print("=" * 50)
    
    # Initialize components
    print("\n1. Initializing agent components...")
    content_analyzer = ContentAnalyzer()
    belief_tracker = BeliefTracker()
    reporter = DailyReporter()
    
    # Create demo data
    print("\n2. Creating demo papers and repositories...")
    papers = create_demo_papers()
    repos = create_demo_repos()
    
    print(f"   Created {len(papers)} demo papers")
    print(f"   Created {len(repos)} demo repositories")
    
    # Analyze content and extract evidence
    print("\n3. Analyzing content and extracting evidence...")
    evidence_list = []
    
    for paper in papers:
        paper_evidence = content_analyzer.analyze_paper(paper)
        evidence_list.extend(paper_evidence)
        print(f"   📄 {paper.title[:50]}... -> {len(paper_evidence)} evidence pieces")
    
    for repo in repos:
        repo_evidence = content_analyzer.analyze_repo(repo)
        evidence_list.extend(repo_evidence)
        print(f"   📦 {repo.name} -> {len(repo_evidence)} evidence pieces")
    
    print(f"\n   Total evidence extracted: {len(evidence_list)}")
    
    # Update beliefs
    print("\n4. Updating Bayesian beliefs...")
    for evidence in evidence_list:
        belief_tracker.update_belief(evidence)
    
    # Show current beliefs
    print("\n5. Current belief state:")
    summary = belief_tracker.generate_summary()
    
    if summary.get("most_promising"):
        print("\n   🚀 Most promising techniques:")
        for i, tech in enumerate(summary["most_promising"][:5], 1):
            belief = belief_tracker.beliefs[tech]
            print(f"      {i}. {tech} (effectiveness: {belief.mean_effectiveness:.3f}, "
                  f"certainty: {belief.certainty:.2f})")
    
    if summary.get("most_overhyped"):
        print("\n   😮‍💨 Most overhyped techniques:")
        for i, tech in enumerate(summary["most_overhyped"][:3], 1):
            belief = belief_tracker.beliefs[tech]
            print(f"      {i}. {tech} (effectiveness: {belief.mean_effectiveness:.3f})")
    
    # Generate report
    print("\n6. Generating daily report...")
    report_data = {
        "papers": papers,
        "repos": repos,
        "evidence": evidence_list,
        "belief_summary": summary,
        "cycle_results": {
            "papers_found": len(papers),
            "repos_found": len(repos),
            "beliefs_updated": len(evidence_list),
            "new_techniques": len(belief_tracker.beliefs)
        }
    }
    
    report_file = reporter.generate_report(report_data)
    
    # Save beliefs
    beliefs_file = belief_tracker.save_beliefs()
    
    print("\n✅ Demo completed successfully!")
    print(f"   📄 Report saved to: {report_file}")
    print(f"   🧠 Beliefs saved to: {beliefs_file}")
    print("\nTry running:")
    print("   python -m src.analysis.view_beliefs")
    print("   to see the belief summary!")


if __name__ == "__main__":
    run_demo() 