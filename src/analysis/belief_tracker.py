"""
Bayesian belief tracking for RL technique effectiveness
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import beta, gamma
from typing import Dict, List, Optional, Tuple
import json
import os
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum


class EvidenceType(Enum):
    """Types of evidence for technique effectiveness"""
    PAPER_RESULT = "paper_result"
    REPO_POPULARITY = "repo_popularity"
    COMMUNITY_ADOPTION = "community_adoption"
    BENCHMARK_SCORE = "benchmark_score"
    EXPERT_OPINION = "expert_opinion"


@dataclass
class Evidence:
    """Represents a piece of evidence about a technique"""
    technique: str
    evidence_type: EvidenceType
    value: float  # 0-1 scale, where 1 is most positive
    confidence: float  # 0-1 scale
    source: str
    timestamp: datetime
    context: Dict = None
    
    def to_dict(self) -> Dict:
        return {
            "technique": self.technique,
            "evidence_type": self.evidence_type.value,
            "value": self.value,
            "confidence": self.confidence,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context or {}
        }


@dataclass
class TechniqueBelief:
    """Bayesian belief about a technique's effectiveness"""
    technique: str
    alpha: float  # Beta distribution parameter (successes)
    beta_param: float  # Beta distribution parameter (failures)
    last_updated: datetime
    evidence_count: int = 0
    
    @property
    def mean_effectiveness(self) -> float:
        """Expected effectiveness (mean of beta distribution)"""
        return self.alpha / (self.alpha + self.beta_param)
    
    @property
    def variance(self) -> float:
        """Variance of effectiveness belief"""
        total = self.alpha + self.beta_param
        return (self.alpha * self.beta_param) / (total**2 * (total + 1))
    
    @property
    def confidence_interval(self, confidence: float = 0.95) -> Tuple[float, float]:
        """Credible interval for effectiveness"""
        dist = beta(self.alpha, self.beta_param)
        tail = (1 - confidence) / 2
        return dist.ppf(tail), dist.ppf(1 - tail)
    
    @property
    def certainty(self) -> float:
        """How certain we are about this belief (0-1)"""
        # Higher alpha + beta means more certainty
        total = self.alpha + self.beta_param
        # Normalize by a reasonable scale (techniques with 100+ evidence points are very certain)
        return min(1.0, total / 100.0)
    
    def to_dict(self) -> Dict:
        return {
            "technique": self.technique,
            "alpha": self.alpha,
            "beta_param": self.beta_param,
            "last_updated": self.last_updated.isoformat(),
            "evidence_count": self.evidence_count,
            "mean_effectiveness": self.mean_effectiveness,
            "variance": self.variance,
            "certainty": self.certainty
        }


class BeliefTracker:
    """Tracks and updates Bayesian beliefs about RL techniques"""
    
    def __init__(self, data_dir: str = "data/beliefs"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        self.beliefs: Dict[str, TechniqueBelief] = {}
        self.evidence_history: List[Evidence] = []
        
        # Prior parameters for new techniques (weakly informative)
        self.prior_alpha = 2.0  # Slightly optimistic prior
        self.prior_beta = 2.0
        
        # Load existing beliefs if available
        self._load_beliefs()
    
    def update_belief(self, evidence: Evidence) -> TechniqueBelief:
        """Update belief about a technique with new evidence"""
        technique = evidence.technique
        
        # Initialize belief if not exists
        if technique not in self.beliefs:
            self.beliefs[technique] = TechniqueBelief(
                technique=technique,
                alpha=self.prior_alpha,
                beta_param=self.prior_beta,
                last_updated=datetime.now(),
                evidence_count=0
            )
        
        belief = self.beliefs[technique]
        
        # Convert evidence to beta distribution updates
        # High value -> more alpha, low value -> more beta
        evidence_weight = evidence.confidence
        
        if evidence.value > 0.5:
            # Positive evidence
            alpha_update = evidence_weight * (evidence.value - 0.5) * 2
            beta_update = evidence_weight * (1 - evidence.value) * 2
        else:
            # Negative evidence  
            alpha_update = evidence_weight * evidence.value * 2
            beta_update = evidence_weight * (0.5 - evidence.value) * 2
        
        # Update belief parameters
        belief.alpha += alpha_update
        belief.beta_param += beta_update
        belief.last_updated = datetime.now()
        belief.evidence_count += 1
        
        # Store evidence
        self.evidence_history.append(evidence)
        
        print(f"Updated belief for {technique}: "
              f"effectiveness={belief.mean_effectiveness:.3f} ± "
              f"{np.sqrt(belief.variance):.3f} "
              f"(certainty={belief.certainty:.2f})")
        
        return belief
    
    def get_technique_ranking(self, min_certainty: float = 0.1) -> List[Tuple[str, float, float]]:
        """Get techniques ranked by effectiveness, filtered by certainty"""
        ranked = []
        
        for technique, belief in self.beliefs.items():
            if belief.certainty >= min_certainty:
                ranked.append((
                    technique,
                    belief.mean_effectiveness,
                    belief.certainty
                ))
        
        # Sort by effectiveness, then by certainty
        ranked.sort(key=lambda x: (x[1], x[2]), reverse=True)
        return ranked
    
    def get_most_promising(self, top_k: int = 10) -> List[str]:
        """Get most promising techniques"""
        ranking = self.get_technique_ranking()
        return [tech for tech, _, _ in ranking[:top_k]]
    
    def get_most_overhyped(self, top_k: int = 10) -> List[str]:
        """Get most overhyped techniques (high certainty, low effectiveness)"""
        overhyped = []
        
        for technique, belief in self.beliefs.items():
            if belief.certainty >= 0.3:  # Need reasonable certainty
                # Overhyped = high certainty but low effectiveness
                overhype_score = belief.certainty * (1 - belief.mean_effectiveness)
                overhyped.append((technique, overhype_score, belief.mean_effectiveness))
        
        overhyped.sort(key=lambda x: x[1], reverse=True)
        return [tech for tech, _, _ in overhyped[:top_k]]
    
    def get_uncertain_techniques(self, top_k: int = 10) -> List[str]:
        """Get techniques we're most uncertain about"""
        uncertain = [(tech, belief.certainty) for tech, belief in self.beliefs.items()]
        uncertain.sort(key=lambda x: x[1])  # Lowest certainty first
        return [tech for tech, _ in uncertain[:top_k]]
    
    def compare_techniques(self, tech1: str, tech2: str) -> Dict:
        """Compare two techniques statistically"""
        if tech1 not in self.beliefs or tech2 not in self.beliefs:
            return {"error": "One or both techniques not found"}
        
        belief1 = self.beliefs[tech1]
        belief2 = self.beliefs[tech2]
        
        # Sample from both distributions
        samples1 = np.random.beta(belief1.alpha, belief1.beta_param, 10000)
        samples2 = np.random.beta(belief2.alpha, belief2.beta_param, 10000)
        
        # Probability that tech1 is better than tech2
        prob_tech1_better = np.mean(samples1 > samples2)
        
        return {
            "technique_1": tech1,
            "technique_2": tech2,
            "mean_effectiveness_1": belief1.mean_effectiveness,
            "mean_effectiveness_2": belief2.mean_effectiveness,
            "certainty_1": belief1.certainty,
            "certainty_2": belief2.certainty,
            "prob_tech1_better": prob_tech1_better,
            "prob_tech2_better": 1 - prob_tech1_better
        }
    
    def save_beliefs(self, filename: Optional[str] = None) -> str:
        """Save current beliefs to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"beliefs_{timestamp}.json"
        
        filepath = os.path.join(self.data_dir, filename)
        
        # Prepare data for JSON serialization
        data = {
            "beliefs": {tech: belief.to_dict() for tech, belief in self.beliefs.items()},
            "evidence_history": [ev.to_dict() for ev in self.evidence_history],
            "metadata": {
                "total_techniques": len(self.beliefs),
                "total_evidence": len(self.evidence_history),
                "last_updated": datetime.now().isoformat()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        return filepath
    
    def _load_beliefs(self):
        """Load most recent beliefs from file"""
        try:
            # Find most recent beliefs file
            belief_files = [f for f in os.listdir(self.data_dir) if f.startswith("beliefs_")]
            if not belief_files:
                return
            
            latest_file = max(belief_files)
            filepath = os.path.join(self.data_dir, latest_file)
            
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Load beliefs
            for tech, belief_data in data["beliefs"].items():
                self.beliefs[tech] = TechniqueBelief(
                    technique=belief_data["technique"],
                    alpha=belief_data["alpha"],
                    beta_param=belief_data["beta_param"],
                    last_updated=datetime.fromisoformat(belief_data["last_updated"]),
                    evidence_count=belief_data["evidence_count"]
                )
            
            # Load evidence history
            for ev_data in data["evidence_history"]:
                evidence = Evidence(
                    technique=ev_data["technique"],
                    evidence_type=EvidenceType(ev_data["evidence_type"]),
                    value=ev_data["value"],
                    confidence=ev_data["confidence"],
                    source=ev_data["source"],
                    timestamp=datetime.fromisoformat(ev_data["timestamp"]),
                    context=ev_data.get("context", {})
                )
                self.evidence_history.append(evidence)
            
            print(f"Loaded {len(self.beliefs)} beliefs and {len(self.evidence_history)} evidence points")
            
        except Exception as e:
            print(f"Could not load previous beliefs: {e}")
    
    def generate_summary(self) -> Dict:
        """Generate summary of current beliefs"""
        if not self.beliefs:
            return {"message": "No beliefs tracked yet"}
        
        ranking = self.get_technique_ranking()
        promising = self.get_most_promising(5)
        overhyped = self.get_most_overhyped(5)
        uncertain = self.get_uncertain_techniques(5)
        
        return {
            "total_techniques": len(self.beliefs),
            "total_evidence": len(self.evidence_history),
            "most_promising": promising,
            "most_overhyped": overhyped,
            "most_uncertain": uncertain,
            "top_10_ranking": [(tech, f"{eff:.3f}") for tech, eff, _ in ranking[:10]]
        }


if __name__ == "__main__":
    # Example usage
    tracker = BeliefTracker()
    
    # Add some example evidence
    evidence1 = Evidence(
        technique="PPO",
        evidence_type=EvidenceType.PAPER_RESULT,
        value=0.8,
        confidence=0.9,
        source="OpenAI Five paper",
        timestamp=datetime.now()
    )
    
    tracker.update_belief(evidence1)
    
    summary = tracker.generate_summary()
    print(json.dumps(summary, indent=2)) 