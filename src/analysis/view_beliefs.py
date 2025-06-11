"""
Simple tool to view current beliefs about RL techniques
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.belief_tracker import BeliefTracker
import json


def main():
    """View current beliefs"""
    tracker = BeliefTracker()
    
    if not tracker.beliefs:
        print("No beliefs tracked yet. Run the daily agent first!")
        return
    
    print("🎯 Current Beliefs About RL Techniques")
    print("=" * 50)
    
    # Generate summary
    summary = tracker.generate_summary()
    
    print(f"\n📊 Overview:")
    print(f"Total techniques tracked: {summary.get('total_techniques', 0)}")
    print(f"Total evidence points: {summary.get('total_evidence', 0)}")
    
    # Most promising
    print(f"\n🚀 Most Promising Techniques:")
    for i, tech in enumerate(summary.get('most_promising', [])[:5], 1):
        belief = tracker.beliefs[tech]
        print(f"{i:2d}. {tech:<20} effectiveness: {belief.mean_effectiveness:.3f} "
              f"(certainty: {belief.certainty:.2f})")
    
    # Most overhyped
    if summary.get('most_overhyped'):
        print(f"\n😮‍💨 Most Overhyped Techniques:")
        for i, tech in enumerate(summary.get('most_overhyped', [])[:3], 1):
            belief = tracker.beliefs[tech]
            print(f"{i:2d}. {tech:<20} effectiveness: {belief.mean_effectiveness:.3f} "
                  f"(certainty: {belief.certainty:.2f})")
    
    # Most uncertain
    if summary.get('most_uncertain'):
        print(f"\n❓ Need More Evidence:")
        for i, tech in enumerate(summary.get('most_uncertain', [])[:3], 1):
            belief = tracker.beliefs[tech]
            print(f"{i:2d}. {tech:<20} effectiveness: {belief.mean_effectiveness:.3f} "
                  f"(certainty: {belief.certainty:.2f})")
    
    # Full ranking
    print(f"\n📈 Full Effectiveness Ranking:")
    ranking = tracker.get_technique_ranking()
    
    print(f"{'Rank':<5} {'Technique':<25} {'Effectiveness':<15} {'Certainty':<10} {'Evidence':<10}")
    print("-" * 70)
    
    for i, (tech, eff, cert) in enumerate(ranking[:15], 1):
        belief = tracker.beliefs[tech]
        print(f"{i:<5} {tech:<25} {eff:<15.3f} {cert:<10.2f} {belief.evidence_count:<10}")
    
    if len(ranking) > 15:
        print(f"... and {len(ranking) - 15} more techniques")
    
    print("\n" + "=" * 50)


if __name__ == "__main__":
    main() 