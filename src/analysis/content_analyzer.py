"""
Content analyzer - extracts evidence about RL techniques from papers and repos
"""

import re
from typing import List, Dict, Optional
from datetime import datetime
import numpy as np

from .belief_tracker import Evidence, EvidenceType
from ..discovery.paper_finder import Paper
from ..discovery.repo_finder import Repository


class ContentAnalyzer:
    """Analyzes papers and repos to extract evidence about technique effectiveness"""
    
    def __init__(self):
        # RL technique keywords and their variants
        self.technique_patterns = {
            "PPO": [r"ppo", r"proximal policy optimization"],
            "SAC": [r"\bsac\b", r"soft actor.?critic"],
            "DDPG": [r"ddpg", r"deep deterministic policy gradient"],
            "DQN": [r"\bdqn\b", r"deep q.?network", r"q.?learning"],
            "A3C": [r"\ba3c\b", r"asynchronous advantage actor.?critic"],
            "TRPO": [r"trpo", r"trust region policy optimization"],
            "Rainbow": [r"rainbow", r"rainbow dqn"],
            "TD3": [r"\btd3\b", r"twin delayed ddpg"],
            "Impala": [r"impala"],
            "Apex": [r"apex", r"distributed prioritized experience replay"],
            "AlphaZero": [r"alphazero", r"alpha zero"],
            "MuZero": [r"muzero", r"mu zero"],
            "Actor-Critic": [r"actor.?critic"],
            "Policy Gradient": [r"policy gradient", r"reinforce"],
            "Monte Carlo Tree Search": [r"mcts", r"monte carlo tree search"],
            "Transformer RL": [r"transformer.*reinforcement", r"decision transformer"],
            "Meta-Learning": [r"meta.?learning", r"learning to learn", r"maml"],
            "Multi-Agent RL": [r"multi.?agent", r"multiagent"],
            "Hierarchical RL": [r"hierarchical.*rl", r"option.*learning"],
            "Offline RL": [r"offline.*rl", r"batch.*rl"],
            "Model-Based RL": [r"model.?based.*rl", r"world model"],
            "Curiosity-Driven": [r"curiosity", r"intrinsic motivation"],
            "Distributional RL": [r"distributional.*rl", r"c51", r"qr.?dqn"]
        }
        
        # Positive/negative indicators in text
        self.positive_indicators = [
            r"state.?of.?the.?art", r"sota", r"outperform", r"improve", r"better",
            r"superior", r"achieve", r"success", r"effective", r"efficient",
            r"breakthrough", r"novel", r"significant improvement", r"surpass"
        ]
        
        self.negative_indicators = [
            r"fail", r"poor", r"worse", r"ineffective", r"unstable",
            r"difficult", r"challenge", r"limitation", r"problem",
            r"struggle", r"inferior", r"underperform"
        ]
        
        # Score keywords for numerical results
        self.score_patterns = [
            r"score[:\s]+(\d+\.?\d*)",
            r"reward[:\s]+(\d+\.?\d*)",
            r"return[:\s]+(\d+\.?\d*)",
            r"performance[:\s]+(\d+\.?\d*)",
            r"accuracy[:\s]+(\d+\.?\d*)%?"
        ]
    
    def analyze_paper(self, paper: Paper) -> List[Evidence]:
        """Extract evidence from a paper"""
        evidence_list = []
        
        # Combine title and abstract for analysis
        text = f"{paper.title} {paper.abstract}".lower()
        
        # Find mentioned techniques
        mentioned_techniques = self._find_techniques(text)
        
        for technique in mentioned_techniques:
            # Analyze sentiment/effectiveness for this technique
            effectiveness_score = self._analyze_technique_effectiveness(text, technique)
            confidence = self._calculate_confidence(paper, technique)
            
            evidence = Evidence(
                technique=technique,
                evidence_type=EvidenceType.PAPER_RESULT,
                value=effectiveness_score,
                confidence=confidence,
                source=f"Paper: {paper.title}",
                timestamp=paper.published_date,
                context={
                    "paper_url": paper.url,
                    "authors": paper.authors,
                    "categories": paper.categories
                }
            )
            evidence_list.append(evidence)
        
        return evidence_list
    
    def analyze_repo(self, repo: Repository) -> List[Evidence]:
        """Extract evidence from a repository"""
        evidence_list = []
        
        # Combine repo name, description for analysis
        text = f"{repo.name} {repo.description}".lower()
        
        # Find mentioned techniques
        mentioned_techniques = self._find_techniques(text)
        
        for technique in mentioned_techniques:
            # Repository popularity as effectiveness indicator
            popularity_score = self._calculate_repo_popularity_score(repo)
            confidence = self._calculate_repo_confidence(repo)
            
            evidence = Evidence(
                technique=technique,
                evidence_type=EvidenceType.REPO_POPULARITY,
                value=popularity_score,
                confidence=confidence,
                source=f"Repository: {repo.full_name}",
                timestamp=repo.updated_at,
                context={
                    "repo_url": repo.url,
                    "stars": repo.stars,
                    "forks": repo.forks,
                    "language": repo.language,
                    "topics": repo.topics
                }
            )
            evidence_list.append(evidence)
        
        return evidence_list
    
    def _find_techniques(self, text: str) -> List[str]:
        """Find RL techniques mentioned in text"""
        found_techniques = []
        
        for technique, patterns in self.technique_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    found_techniques.append(technique)
                    break  # Only add once per technique
        
        return found_techniques
    
    def _analyze_technique_effectiveness(self, text: str, technique: str) -> float:
        """Analyze effectiveness sentiment for a technique in text"""
        # Count positive and negative indicators
        positive_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) 
                           for pattern in self.positive_indicators)
        negative_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) 
                           for pattern in self.negative_indicators)
        
        # Look for numerical scores
        numerical_score = self._extract_numerical_score(text)
        
        # Combine sentiment and numerical evidence
        if numerical_score is not None:
            # If we have numerical scores, weight them heavily
            base_score = numerical_score
        else:
            # Use sentiment analysis
            if positive_count + negative_count == 0:
                base_score = 0.5  # Neutral
            else:
                sentiment_ratio = positive_count / (positive_count + negative_count)
                base_score = sentiment_ratio
        
        # Adjust based on context
        if "state of the art" in text.lower() or "sota" in text.lower():
            base_score = min(1.0, base_score + 0.2)
        
        if "baseline" in text.lower() and technique.lower() in text.lower():
            base_score = max(0.0, base_score - 0.1)
        
        return np.clip(base_score, 0.0, 1.0)
    
    def _extract_numerical_score(self, text: str) -> Optional[float]:
        """Extract numerical performance scores from text"""
        scores = []
        
        for pattern in self.score_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    score = float(match)
                    # Normalize common score ranges
                    if score > 100:  # Likely percentage > 100% or large reward
                        normalized = min(1.0, score / 1000)  # Scale down large rewards
                    elif score > 1:  # Likely percentage or medium reward
                        normalized = min(1.0, score / 100)
                    else:  # Already 0-1 range
                        normalized = score
                    scores.append(normalized)
                except ValueError:
                    continue
        
        if scores:
            return np.mean(scores)  # Average if multiple scores found
        return None
    
    def _calculate_confidence(self, paper: Paper, technique: str) -> float:
        """Calculate confidence in paper evidence"""
        confidence = 0.5  # Base confidence
        
        # More authors generally means more credible
        if len(paper.authors) >= 3:
            confidence += 0.1
        elif len(paper.authors) >= 5:
            confidence += 0.2
        
        # Recent papers are more relevant
        days_old = (datetime.now() - paper.published_date).days
        if days_old <= 30:
            confidence += 0.2
        elif days_old <= 90:
            confidence += 0.1
        
        # Venue quality (rough heuristic based on categories)
        if any(cat in ["cs.LG", "cs.AI"] for cat in paper.categories):
            confidence += 0.1
        
        # Length of abstract suggests thoroughness
        if len(paper.abstract) > 1000:
            confidence += 0.1
        
        return np.clip(confidence, 0.0, 1.0)
    
    def _calculate_repo_popularity_score(self, repo: Repository) -> float:
        """Convert repository metrics to effectiveness score"""
        # Logarithmic scaling for stars (popular repos are more effective)
        if repo.stars <= 0:
            star_score = 0.0
        else:
            # Scale: 1 star = 0.1, 10 stars = 0.3, 100 stars = 0.5, 1000+ stars = 0.8+
            star_score = min(0.8, np.log10(repo.stars + 1) / 4)
        
        # Fork ratio indicates active use
        if repo.stars > 0:
            fork_ratio = repo.forks / repo.stars
            fork_score = min(0.2, fork_ratio * 2)  # Up to 0.2 bonus for high fork ratio
        else:
            fork_score = 0.0
        
        # Recent activity bonus
        days_since_update = (datetime.now() - repo.updated_at).days
        if days_since_update <= 7:
            recency_score = 0.1
        elif days_since_update <= 30:
            recency_score = 0.05
        else:
            recency_score = 0.0
        
        total_score = star_score + fork_score + recency_score
        return np.clip(total_score, 0.0, 1.0)
    
    def _calculate_repo_confidence(self, repo: Repository) -> float:
        """Calculate confidence in repository evidence"""
        confidence = 0.3  # Lower base confidence than papers
        
        # More stars = more confidence
        if repo.stars >= 100:
            confidence += 0.3
        elif repo.stars >= 50:
            confidence += 0.2
        elif repo.stars >= 10:
            confidence += 0.1
        
        # Good description suggests quality
        if repo.description and len(repo.description) > 50:
            confidence += 0.1
        
        # Multiple languages suggest complexity/real use
        if repo.language and repo.language in ["Python", "C++", "JavaScript"]:
            confidence += 0.1
        
        # Recent activity
        days_since_update = (datetime.now() - repo.updated_at).days
        if days_since_update <= 30:
            confidence += 0.1
        
        return np.clip(confidence, 0.0, 1.0) 