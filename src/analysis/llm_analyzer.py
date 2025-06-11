"""
LLM-based content analyzer for RL technique effectiveness assessment
"""

import os
import json
import re
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import requests
import time

from .belief_tracker import Evidence, EvidenceType
from ..discovery.paper_finder import Paper
from ..discovery.repo_finder import Repository


class LLMAnalyzer:
    """LLM-powered analyzer for extracting RL technique effectiveness evidence"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable.")
        
        self.api_base = "https://api.openai.com/v1/chat/completions"
        self.model = "gpt-4o-mini"  # Cost-effective model
        
        # Known RL techniques to look for
        self.rl_techniques = [
            "PPO", "SAC", "DDPG", "DQN", "A3C", "TRPO", "Rainbow", "TD3",
            "Impala", "Apex", "AlphaZero", "MuZero", "Actor-Critic",
            "Policy Gradient", "Monte Carlo Tree Search", "Transformer RL",
            "Meta-Learning", "Multi-Agent RL", "Hierarchical RL", "Offline RL",
            "Model-Based RL", "Curiosity-Driven", "Distributional RL"
        ]
    
    def analyze_paper(self, paper: Paper) -> List[Evidence]:
        """Extract evidence from a paper using LLM"""
        evidence_list = []
        
        # Prepare paper content for LLM
        paper_content = f"""
        Title: {paper.title}
        Abstract: {paper.abstract}
        Authors: {', '.join(paper.authors)}
        Categories: {', '.join(paper.categories)}
        """
        
        try:
            # Get LLM analysis
            analysis = self._analyze_with_llm(paper_content, "paper")
            
            # Convert LLM response to evidence objects
            for technique_data in analysis.get("techniques", []):
                confidence = self._calculate_paper_confidence(paper, technique_data)
                
                evidence = Evidence(
                    technique=technique_data["name"],
                    evidence_type=EvidenceType.PAPER_RESULT,
                    value=technique_data["effectiveness_score"],
                    confidence=confidence,
                    source=f"Paper: {paper.title}",
                    timestamp=paper.published_date,
                    context={
                        "paper_url": paper.url,
                        "authors": paper.authors,
                        "categories": paper.categories,
                        "llm_reasoning": technique_data.get("reasoning", ""),
                        "llm_confidence": technique_data.get("confidence", 0.5)
                    }
                )
                evidence_list.append(evidence)
                
        except Exception as e:
            print(f"Error analyzing paper {paper.title}: {e}")
        
        return evidence_list
    
    def analyze_repo(self, repo: Repository) -> List[Evidence]:
        """Extract evidence from a repository using LLM"""
        evidence_list = []
        
        # Prepare repo content for LLM
        repo_content = f"""
        Name: {repo.name}
        Description: {repo.description}
        Language: {repo.language}
        Stars: {repo.stars}
        Forks: {repo.forks}
        Topics: {', '.join(repo.topics)}
        Last Updated: {repo.updated_at}
        """
        
        try:
            # Get LLM analysis
            analysis = self._analyze_with_llm(repo_content, "repository")
            
            # Convert LLM response to evidence objects
            for technique_data in analysis.get("techniques", []):
                # Combine LLM assessment with popularity metrics
                popularity_score = self._calculate_repo_popularity_score(repo)
                llm_score = technique_data["effectiveness_score"]
                
                # Weight: 60% LLM assessment, 40% popularity
                combined_score = 0.6 * llm_score + 0.4 * popularity_score
                
                confidence = self._calculate_repo_confidence(repo, technique_data)
                
                evidence = Evidence(
                    technique=technique_data["name"],
                    evidence_type=EvidenceType.REPO_POPULARITY,
                    value=combined_score,
                    confidence=confidence,
                    source=f"Repository: {repo.full_name}",
                    timestamp=repo.updated_at,
                    context={
                        "repo_url": repo.url,
                        "stars": repo.stars,
                        "forks": repo.forks,
                        "language": repo.language,
                        "topics": repo.topics,
                        "llm_reasoning": technique_data.get("reasoning", ""),
                        "llm_confidence": technique_data.get("confidence", 0.5),
                        "popularity_score": popularity_score
                    }
                )
                evidence_list.append(evidence)
                
        except Exception as e:
            print(f"Error analyzing repo {repo.full_name}: {e}")
        
        return evidence_list
    
    def _analyze_with_llm(self, content: str, content_type: str) -> Dict:
        """Send content to LLM for analysis"""
        
        prompt = f"""
        You are an expert in reinforcement learning research. Analyze the following {content_type} and extract information about RL techniques mentioned.

        {content_type.upper()} CONTENT:
        {content}

        TASK:
        For each RL technique mentioned, assess:
        1. Effectiveness score (0.0-1.0): How effective/successful is this technique based on the content?
        2. Confidence (0.0-1.0): How confident are you in this assessment?
        3. Reasoning: Brief explanation of your assessment

        SCORING GUIDELINES:
        - 0.9-1.0: Breakthrough results, state-of-the-art performance, significantly outperforms baselines
        - 0.7-0.9: Strong positive results, clearly effective, outperforms most alternatives
        - 0.5-0.7: Moderate effectiveness, some advantages, mixed results
        - 0.3-0.5: Limited effectiveness, significant limitations identified
        - 0.0-0.3: Poor performance, major issues, fails to work well

        KNOWN RL TECHNIQUES:
        {', '.join(self.rl_techniques)}

        Return JSON format:
        {{
            "techniques": [
                {{
                    "name": "PPO",
                    "effectiveness_score": 0.85,
                    "confidence": 0.9,
                    "reasoning": "Paper shows PPO achieved superhuman performance on Dota 2, outperforming previous methods significantly"
                }}
            ]
        }}

        Only include techniques explicitly mentioned or clearly implied in the content.
        """
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 1000
        }
        
        response = requests.post(self.api_base, headers=headers, json=data)
        response.raise_for_status()
        
        result = response.json()
        content_text = result["choices"][0]["message"]["content"]
        
        # Parse JSON response
        try:
            # Extract JSON from response (handle cases where LLM adds extra text)
            json_match = re.search(r'\{.*\}', content_text, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group())
            else:
                analysis = json.loads(content_text)
            
            return analysis
            
        except json.JSONDecodeError as e:
            print(f"Error parsing LLM response: {e}")
            print(f"Response was: {content_text}")
            return {"techniques": []}
    
    def _calculate_paper_confidence(self, paper: Paper, technique_data: Dict) -> float:
        """Calculate confidence for paper evidence"""
        confidence = 0.5  # Base confidence
        
        # Author count (more authors = slightly more reliable)
        author_bonus = min(0.1, len(paper.authors) * 0.02)
        confidence += author_bonus
        
        # Recent papers are more relevant
        days_old = (datetime.now() - paper.published_date).days
        if days_old <= 30:
            confidence += 0.2
        elif days_old <= 90:
            confidence += 0.1
        
        # LLM confidence
        llm_conf = technique_data.get("confidence", 0.5)
        confidence = 0.7 * confidence + 0.3 * llm_conf
        
        # Abstract length (longer abstracts often more detailed)
        if len(paper.abstract) > 1000:
            confidence += 0.05
        
        return min(1.0, confidence)
    
    def _calculate_repo_confidence(self, repo: Repository, technique_data: Dict) -> float:
        """Calculate confidence for repository evidence"""
        confidence = 0.3  # Lower base for repos
        
        # Star count
        if repo.stars > 1000:
            confidence += 0.3
        elif repo.stars > 100:
            confidence += 0.2
        elif repo.stars > 10:
            confidence += 0.1
        
        # Good description
        if repo.description and len(repo.description) > 50:
            confidence += 0.1
        
        # Popular language
        if repo.language in ["Python", "C++", "JavaScript"]:
            confidence += 0.05
        
        # Recent activity
        days_since_update = (datetime.now() - repo.updated_at).days
        if days_since_update <= 30:
            confidence += 0.15
        elif days_since_update <= 90:
            confidence += 0.1
        
        # LLM confidence
        llm_conf = technique_data.get("confidence", 0.5)
        confidence = 0.6 * confidence + 0.4 * llm_conf
        
        return min(1.0, confidence)
    
    def _calculate_repo_popularity_score(self, repo: Repository) -> float:
        """Calculate popularity-based effectiveness score"""
        # Logarithmic scaling for stars
        star_score = min(0.8, np.log10(repo.stars + 1) / 4)
        
        # Fork ratio bonus
        if repo.stars > 0:
            fork_ratio = repo.forks / repo.stars
            fork_bonus = min(0.2, fork_ratio * 2)
        else:
            fork_bonus = 0
        
        # Recency bonus
        days_since_update = (datetime.now() - repo.updated_at).days
        if days_since_update <= 7:
            recency_bonus = 0.1
        elif days_since_update <= 30:
            recency_bonus = 0.05
        else:
            recency_bonus = 0
        
        return min(1.0, star_score + fork_bonus + recency_bonus)


# Import numpy for calculations
import numpy as np 