"""
Paper discovery module - finds new RL papers from various sources
"""

import arxiv
import requests
import feedparser
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import json
import os
from dataclasses import dataclass
import time


@dataclass
class Paper:
    """Represents a discovered paper"""
    title: str
    authors: List[str]
    abstract: str
    url: str
    published_date: datetime
    categories: List[str]
    source: str = "arxiv"
    
    def to_dict(self) -> Dict:
        return {
            "title": self.title,
            "authors": self.authors,
            "abstract": self.abstract,
            "url": self.url,
            "published_date": self.published_date.isoformat(),
            "categories": self.categories,
            "source": self.source
        }


class PaperFinder:
    """Discovers new RL papers from multiple sources"""
    
    def __init__(self, data_dir: str = "data/papers"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # RL-related keywords and categories
        self.rl_keywords = [
            "reinforcement learning", "deep reinforcement learning", 
            "policy gradient", "q-learning", "actor-critic",
            "bayesian reinforcement learning", "exploration",
            "multi-agent reinforcement learning", "meta-learning rl",
            "model-based reinforcement learning"
        ]
        
        self.arxiv_categories = [
            "cs.LG", "cs.AI", "stat.ML", "cs.RO"  # machine learning, AI, robotics
        ]

    def find_arxiv_papers(self, days_back: int = 7) -> List[Paper]:
        """Find recent RL papers on ArXiv"""
        papers = []
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        # Search for recent papers in relevant categories
        for category in self.arxiv_categories:
            try:
                search = arxiv.Search(
                    query=f'cat:{category}',
                    max_results=100,
                    sort_by=arxiv.SortCriterion.SubmittedDate
                )
                
                for result in search.results():
                    # Filter for RL-related content
                    if self._is_rl_related(result.title, result.summary):
                        if result.published >= cutoff_date:
                            paper = Paper(
                                title=result.title,
                                authors=[str(author) for author in result.authors],
                                abstract=result.summary,
                                url=result.entry_id,
                                published_date=result.published,
                                categories=[category],
                                source="arxiv"
                            )
                            papers.append(paper)
                
                # Be nice to ArXiv
                time.sleep(1)
                
            except Exception as e:
                print(f"Error searching ArXiv category {category}: {e}")
                continue
        
        return papers

    def find_conference_papers(self) -> List[Paper]:
        """Find papers from recent major conferences (RSS feeds where available)"""
        papers = []
        
        # Major RL conferences and their RSS feeds (when available)
        conference_feeds = {
            "NeurIPS": "https://papers.nips.cc/rss",  # hypothetical
            "ICML": "https://icml.cc/rss",  # hypothetical
        }
        
        for conf_name, feed_url in conference_feeds.items():
            try:
                # This would need to be customized per conference
                # For now, just a placeholder structure
                pass
            except Exception as e:
                print(f"Error fetching {conf_name} papers: {e}")
        
        return papers

    def _is_rl_related(self, title: str, abstract: str) -> bool:
        """Check if paper is RL-related based on keywords"""
        text = f"{title} {abstract}".lower()
        
        return any(keyword in text for keyword in self.rl_keywords)

    def save_papers(self, papers: List[Paper], filename: Optional[str] = None) -> str:
        """Save discovered papers to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"papers_{timestamp}.json"
        
        filepath = os.path.join(self.data_dir, filename)
        
        papers_data = [paper.to_dict() for paper in papers]
        
        with open(filepath, 'w') as f:
            json.dump(papers_data, f, indent=2, default=str)
        
        return filepath

    def load_papers(self, filename: str) -> List[Paper]:
        """Load papers from JSON file"""
        filepath = os.path.join(self.data_dir, filename)
        
        with open(filepath, 'r') as f:
            papers_data = json.load(f)
        
        papers = []
        for data in papers_data:
            paper = Paper(
                title=data["title"],
                authors=data["authors"],
                abstract=data["abstract"],
                url=data["url"],
                published_date=datetime.fromisoformat(data["published_date"]),
                categories=data["categories"],
                source=data.get("source", "unknown")
            )
            papers.append(paper)
        
        return papers

    def discover_daily(self) -> List[Paper]:
        """Main discovery method - find new papers from all sources"""
        print("Starting daily paper discovery...")
        
        all_papers = []
        
        # ArXiv papers
        print("Searching ArXiv...")
        arxiv_papers = self.find_arxiv_papers(days_back=2)
        all_papers.extend(arxiv_papers)
        print(f"Found {len(arxiv_papers)} ArXiv papers")
        
        # Conference papers
        print("Searching conferences...")
        conf_papers = self.find_conference_papers()
        all_papers.extend(conf_papers)
        print(f"Found {len(conf_papers)} conference papers")
        
        # Remove duplicates based on title similarity
        unique_papers = self._deduplicate_papers(all_papers)
        
        print(f"Total unique papers found: {len(unique_papers)}")
        
        return unique_papers

    def _deduplicate_papers(self, papers: List[Paper]) -> List[Paper]:
        """Remove duplicate papers based on title similarity"""
        seen_titles = set()
        unique_papers = []
        
        for paper in papers:
            # Simple deduplication - could be improved with fuzzy matching
            title_lower = paper.title.lower().strip()
            if title_lower not in seen_titles:
                seen_titles.add(title_lower)
                unique_papers.append(paper)
        
        return unique_papers


if __name__ == "__main__":
    finder = PaperFinder()
    papers = finder.discover_daily()
    saved_file = finder.save_papers(papers)
    print(f"Saved {len(papers)} papers to {saved_file}") 