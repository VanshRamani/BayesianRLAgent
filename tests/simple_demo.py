#!/usr/bin/env python3
"""
Simplified demo for Bayesian RL Agent - shows core belief tracking without external APIs
"""

import sys
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Add src to path
sys.path.append('src')

from analysis.belief_tracker import BeliefTracker, Evidence, EvidenceType
import json

load_dotenv()


def run_simple_demo():
    """Run a simplified demo showing belief tracking"""
    print("🚀 Bayesian RL Agent - Simple Demo")
    print("=" * 50)
    print("Demonstrating core Bayesian belief tracking...")
    
    # Initialize belief tracker
    print("\n1. Initializing belief tracker...")
    tracker = BeliefTracker()
    
    # Simulate evidence from various sources
    print("\n2. Adding evidence about RL techniques...")
    
    evidence_data = [
        # PPO evidence - mostly positive
        ("PPO", 0.85, 0.9, "Paper: OpenAI Five - PPO achieved superhuman performance"),
        ("PPO", 0.75, 0.8, "Repository: stable-baselines3 - 4500 stars, actively maintained"),
        ("PPO", 0.8, 0.85, "Paper: Learning to summarize - PPO outperformed baselines"),
        
        # SAC evidence - positive but mixed
        ("SAC", 0.78, 0.85, "Paper: Soft Actor-Critic achieves state-of-the-art performance"),
        ("SAC", 0.65, 0.7, "Repository: SAC implementation - good but complex"),
        
        # DQN evidence - mixed (shows challenges)
        ("DQN", 0.45, 0.8, "Paper: Challenges in DQN - stability issues identified"),
        ("DQN", 0.6, 0.75, "Repository: Rainbow DQN - 1200 stars but dated"),
        ("DQN", 0.4, 0.7, "Paper: DQN limitations - poor sample efficiency"),
        
        # TD3 evidence - positive
        ("TD3", 0.72, 0.8, "Paper: Twin Delayed DDPG - addresses overestimation bias"),
        ("TD3", 0.68, 0.75, "Repository: TD3 implementation in stable-baselines3"),
        
        # Rainbow evidence - moderate
        ("Rainbow", 0.65, 0.7, "Paper: Rainbow DQN combines multiple improvements"),
        ("Rainbow", 0.58, 0.65, "Repository: Implementation exists but limited adoption"),
        
        # A3C evidence - older technique, mixed results
        ("A3C", 0.55, 0.6, "Paper: A3C good for distributed training"),
        ("A3C", 0.45, 0.65, "Repository: A3C implementation - less popular now"),
    ]
    
    for technique, value, confidence, source in evidence_data:
        evidence = Evidence(
            technique=technique,
            evidence_type=EvidenceType.PAPER_RESULT if "Paper:" in source else EvidenceType.REPO_POPULARITY,
            value=value,
            confidence=confidence,
            source=source,
            timestamp=datetime.now()
        )
        tracker.update_belief(evidence)
    
    print(f"\n   Total evidence processed: {len(evidence_data)}")
    
    # Show results
    print("\n3. Current belief state:")
    summary = tracker.generate_summary()
    
    print(f"\n📊 Overview:")
    print(f"   Total techniques tracked: {summary['total_techniques']}")
    print(f"   Total evidence points: {summary['total_evidence']}")
    
    if summary.get("most_promising"):
        print(f"\n🚀 Most promising techniques:")
        for i, tech in enumerate(summary["most_promising"][:5], 1):
            belief = tracker.beliefs[tech]
            ci_low, ci_high = belief.confidence_interval(0.95)
            print(f"   {i}. {tech:<12} effectiveness: {belief.mean_effectiveness:.3f} "
                  f"[{ci_low:.3f}-{ci_high:.3f}] (certainty: {belief.certainty:.2f})")
    
    if summary.get("most_overhyped"):
        print(f"\n😮‍💨 Techniques needing more evidence:")
        for i, tech in enumerate(summary["most_overhyped"][:3], 1):
            belief = tracker.beliefs[tech]
            print(f"   {i}. {tech:<12} effectiveness: {belief.mean_effectiveness:.3f} "
                  f"(certainty: {belief.certainty:.2f})")
    
    # Show technique comparison
    print(f"\n⚖️  Technique Comparisons:")
    comparisons = [
        ("PPO", "SAC"),
        ("PPO", "DQN"),
        ("TD3", "DQN")
    ]
    
    for tech1, tech2 in comparisons:
        if tech1 in tracker.beliefs and tech2 in tracker.beliefs:
            comp = tracker.compare_techniques(tech1, tech2)
            prob = comp["prob_tech1_better"]
            print(f"   {tech1} vs {tech2}: {prob:.1%} chance {tech1} is better")
    
    # Full ranking
    print(f"\n📈 Full Effectiveness Ranking:")
    ranking = tracker.get_technique_ranking()
    
    print(f"{'Rank':<5} {'Technique':<12} {'Effectiveness':<15} {'Certainty':<10} {'Evidence':<10}")
    print("-" * 60)
    
    for i, (tech, eff, cert) in enumerate(ranking, 1):
        belief = tracker.beliefs[tech]
        print(f"{i:<5} {tech:<12} {eff:<15.3f} {cert:<10.2f} {belief.evidence_count:<10}")
    
    # Save beliefs
    beliefs_file = tracker.save_beliefs()
    
    print(f"\n✅ Demo completed successfully!")
    print(f"   🧠 Beliefs saved to: {beliefs_file}")
    print(f"\n💡 Key Insights:")
    print(f"   - PPO shows consistently high effectiveness with good evidence")
    print(f"   - DQN has challenges but this is well-documented")
    print(f"   - SAC is promising but needs more evidence")
    print(f"   - The system tracks uncertainty and adjusts confidence accordingly")
    
    print(f"\n🔬 Bayesian Method:")
    print(f"   - Each technique modeled as Beta distribution")
    print(f"   - Evidence updates α and β parameters")
    print(f"   - Uncertainty decreases as evidence accumulates")
    print(f"   - Confidence intervals show credible ranges")


if __name__ == "__main__":
    run_simple_demo() 