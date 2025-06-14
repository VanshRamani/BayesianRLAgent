#!/usr/bin/env python3
"""
Test Bayesian belief tracking functionality
"""

import sys
import os
from dotenv import load_dotenv
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from analysis.belief_tracker import BeliefTracker, Evidence, EvidenceType
from datetime import datetime
import numpy as np

load_dotenv()

def test_belief_tracking():
    """Test Bayesian belief tracking with sample evidence"""
    print("🧠 Testing Bayesian Belief Tracking...")
    
    try:
        # Initialize tracker (clear state for clean test)
        tracker = BeliefTracker()
        
        # Clear existing beliefs for clean test
        tracker.beliefs = {}
        tracker.evidence_history = []
        
        # Create sample evidence
        evidence_samples = [
            Evidence(
                technique="TEST_PPO",
                evidence_type=EvidenceType.PAPER_RESULT,
                value=0.85,
                confidence=0.9,
                source="Test Paper 1",
                timestamp=datetime.now(),
                context={"reasoning": "High performance in test environment"}
            ),
            Evidence(
                technique="TEST_PPO",
                evidence_type=EvidenceType.PAPER_RESULT,
                value=0.78,
                confidence=0.8,
                source="Test Paper 2",
                timestamp=datetime.now(),
                context={"reasoning": "Good but not exceptional performance"}
            ),
            Evidence(
                technique="TEST_DQN",
                evidence_type=EvidenceType.PAPER_RESULT,
                value=0.65,
                confidence=0.7,
                source="Test Paper 3",
                timestamp=datetime.now(),
                context={"reasoning": "Moderate performance with stability issues"}
            )
        ]
        
        print(f"📊 Adding {len(evidence_samples)} pieces of evidence...")
        
        # Update beliefs
        for evidence in evidence_samples:
            tracker.update_belief(evidence)
            belief = tracker.beliefs[evidence.technique]
            print(f"   ✅ {evidence.technique}: effectiveness={belief.mean_effectiveness:.3f}, "
                  f"variance={belief.variance:.3f}")
        
        # Test belief properties
        print(f"\n🔍 Testing belief properties...")
        for technique in tracker.beliefs.keys():
            belief = tracker.beliefs[technique]
            ci_low, ci_high = belief.confidence_interval  # Property, not method
            
            print(f"\n   {technique}:")
            print(f"     • Effectiveness: {belief.mean_effectiveness:.3f}")
            print(f"     • Variance: {belief.variance:.3f}")
            print(f"     • Std Dev: {np.sqrt(belief.variance):.3f}")
            print(f"     • Certainty: {belief.certainty:.3f}")
            print(f"     • 95% CI: [{ci_low:.3f}, {ci_high:.3f}]")
            print(f"     • Evidence count: {belief.evidence_count}")
            print(f"     • Alpha: {belief.alpha:.3f}, Beta: {belief.beta_param:.3f}")
        
        # Test comparisons
        print(f"\n⚖️  Testing technique comparisons...")
        if "TEST_PPO" in tracker.beliefs and "TEST_DQN" in tracker.beliefs:
            comparison = tracker.compare_techniques("TEST_PPO", "TEST_DQN")
            print(f"   TEST_PPO vs TEST_DQN: {comparison['prob_tech1_better']:.1%} chance TEST_PPO is better")
            print(f"   TEST_PPO effectiveness: {comparison['mean_effectiveness_1']:.3f}")
            print(f"   TEST_DQN effectiveness: {comparison['mean_effectiveness_2']:.3f}")
        
        # Test ranking
        print(f"\n📈 Testing technique ranking...")
        ranking = tracker.get_technique_ranking()
        for i, (tech, eff, cert) in enumerate(ranking[:5], 1):  # Show top 5
            print(f"   {i}. {tech}: effectiveness={eff:.3f}, certainty={cert:.3f}")
        
        # Test summary
        print(f"\n📋 Testing summary generation...")
        summary = tracker.generate_summary()
        print(f"   Total techniques: {summary['total_techniques']}")
        print(f"   Total evidence: {summary['total_evidence']}")
        
        if summary.get("most_promising"):
            print(f"   Most promising: {summary['most_promising'][:3]}")
        
        if summary.get("most_overhyped"):
            print(f"   Most overhyped: {summary['most_overhyped'][:3]}")
        
        print(f"\n✅ All belief tracking tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Belief tracking test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_belief_tracking()
    if success:
        print(f"\n🎉 Bayesian belief system working correctly! 📊")
    else:
        print("💥 Belief tracking test failed") 