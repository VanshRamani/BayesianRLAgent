"""
GitHub repository discovery - finds trending/interesting RL repos
"""

from github import Github
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import json
import os
from dataclasses import dataclass
import time


@dataclass 
class Repository:
    """Represents a discovered GitHub repository"""
    name: str
    full_name: str
    description: str
    url: str
    stars: int
    forks: int
    language: str
    created_at: datetime
    updated_at: datetime
    topics: List[str]
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "full_name": self.full_name,
            "description": self.description,
            "url": self.url,
            "stars": self.stars,
            "forks": self.forks,
            "language": self.language,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "topics": self.topics
        }


class RepoFinder:
    """Discovers interesting RL repositories on GitHub"""
    
    def __init__(self, github_token: Optional[str] = None, data_dir: str = "data/repos"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialize GitHub client
        if github_token:
            self.github = Github(github_token)
        else:
            self.github = Github()  # Anonymous access (rate limited)
        
        # RL-related search terms
        self.rl_topics = [
            "reinforcement-learning",
            "deep-reinforcement-learning", 
            "rl",
            "drl",
            "policy-gradient",
            "q-learning",
            "actor-critic",
            "ddpg",
            "ppo",
            "sac",
            "rainbow",
            "a3c",
            "multi-agent-rl"
        ]
        
        self.rl_keywords = [
            "reinforcement learning",
            "deep rl",
            "policy gradient",
            "q-learning",
            "actor critic",
            "gym environment",
            "openai gym",
            "stable baselines"
        ]

    def find_trending_repos(self, days_back: int = 30) -> List[Repository]:
        """Find trending RL repositories"""
        repos = []
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        # Search by topics
        for topic in self.rl_topics:
            try:
                query = f"topic:{topic} created:>{cutoff_date.strftime('%Y-%m-%d')}"
                results = self.github.search_repositories(
                    query=query,
                    sort="stars",
                    order="desc"
                )
                
                for repo in results:
                    if repo.stargazers_count >= 5:  # Minimum threshold
                        repository = Repository(
                            name=repo.name,
                            full_name=repo.full_name,
                            description=repo.description or "",
                            url=repo.html_url,
                            stars=repo.stargazers_count,
                            forks=repo.forks_count,
                            language=repo.language or "Unknown",
                            created_at=repo.created_at,
                            updated_at=repo.updated_at,
                            topics=repo.get_topics()
                        )
                        repos.append(repository)
                
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                print(f"Error searching topic {topic}: {e}")
                continue
        
        return repos

    def find_popular_repos(self, min_stars: int = 100) -> List[Repository]:
        """Find popular RL repositories with substantial following"""
        repos = []
        
        # Search for established repos
        try:
            query = f"reinforcement learning stars:>{min_stars}"
            results = self.github.search_repositories(
                query=query,
                sort="stars", 
                order="desc"
            )
            
            for repo in results:
                if self._is_rl_related(repo.name, repo.description):
                    repository = Repository(
                        name=repo.name,
                        full_name=repo.full_name,
                        description=repo.description or "",
                        url=repo.html_url,
                        stars=repo.stargazers_count,
                        forks=repo.forks_count,
                        language=repo.language or "Unknown",
                        created_at=repo.created_at,
                        updated_at=repo.updated_at,
                        topics=repo.get_topics()
                    )
                    repos.append(repository)
                    
                    if len(repos) >= 50:  # Limit to avoid rate limits
                        break
            
        except Exception as e:
            print(f"Error searching popular repos: {e}")
        
        return repos

    def find_recently_updated(self, days_back: int = 7) -> List[Repository]:
        """Find recently updated RL repositories"""
        repos = []
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        try:
            query = f"reinforcement learning pushed:>{cutoff_date.strftime('%Y-%m-%d')}"
            results = self.github.search_repositories(
                query=query,
                sort="updated",
                order="desc"
            )
            
            for repo in results:
                if repo.stargazers_count >= 10:  # Some activity threshold
                    repository = Repository(
                        name=repo.name,
                        full_name=repo.full_name,
                        description=repo.description or "",
                        url=repo.html_url,
                        stars=repo.stargazers_count,
                        forks=repo.forks_count,
                        language=repo.language or "Unknown",
                        created_at=repo.created_at,
                        updated_at=repo.updated_at,
                        topics=repo.get_topics()
                    )
                    repos.append(repository)
                    
                    if len(repos) >= 30:
                        break
            
        except Exception as e:
            print(f"Error searching recently updated repos: {e}")
        
        return repos

    def _is_rl_related(self, name: str, description: str) -> bool:
        """Check if repository is RL-related"""
        text = f"{name} {description}".lower()
        return any(keyword in text for keyword in self.rl_keywords)

    def save_repos(self, repos: List[Repository], filename: Optional[str] = None) -> str:
        """Save discovered repositories to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"repos_{timestamp}.json"
        
        filepath = os.path.join(self.data_dir, filename)
        
        repos_data = [repo.to_dict() for repo in repos]
        
        with open(filepath, 'w') as f:
            json.dump(repos_data, f, indent=2, default=str)
        
        return filepath

    def load_repos(self, filename: str) -> List[Repository]:
        """Load repositories from JSON file"""
        filepath = os.path.join(self.data_dir, filename)
        
        with open(filepath, 'r') as f:
            repos_data = json.load(f)
        
        repos = []
        for data in repos_data:
            repo = Repository(
                name=data["name"],
                full_name=data["full_name"],
                description=data["description"],
                url=data["url"],
                stars=data["stars"],
                forks=data["forks"],
                language=data["language"],
                created_at=datetime.fromisoformat(data["created_at"]),
                updated_at=datetime.fromisoformat(data["updated_at"]),
                topics=data["topics"]
            )
            repos.append(repo)
        
        return repos

    def discover_daily(self) -> List[Repository]:
        """Main discovery method - find interesting repos from all sources"""
        print("Starting daily repository discovery...")
        
        all_repos = []
        
        # Trending repos
        print("Finding trending repos...")
        trending = self.find_trending_repos(days_back=30)
        all_repos.extend(trending)
        print(f"Found {len(trending)} trending repos")
        
        # Recently updated repos
        print("Finding recently updated repos...")
        updated = self.find_recently_updated(days_back=7)
        all_repos.extend(updated)
        print(f"Found {len(updated)} recently updated repos")
        
        # Remove duplicates
        unique_repos = self._deduplicate_repos(all_repos)
        
        print(f"Total unique repos found: {len(unique_repos)}")
        
        return unique_repos

    def _deduplicate_repos(self, repos: List[Repository]) -> List[Repository]:
        """Remove duplicate repositories based on full_name"""
        seen_names = set()
        unique_repos = []
        
        for repo in repos:
            if repo.full_name not in seen_names:
                seen_names.add(repo.full_name)
                unique_repos.append(repo)
        
        return unique_repos

    def get_repo_metrics_history(self, repo_full_name: str) -> Dict:
        """Get historical metrics for a repository (simplified version)"""
        try:
            repo = self.github.get_repo(repo_full_name)
            return {
                "current_stars": repo.stargazers_count,
                "current_forks": repo.forks_count,
                "current_issues": repo.open_issues_count,
                "last_commit": repo.updated_at.isoformat(),
                "topics": repo.get_topics()
            }
        except Exception as e:
            print(f"Error getting metrics for {repo_full_name}: {e}")
            return {}


if __name__ == "__main__":
    # Example usage - set GITHUB_TOKEN environment variable for higher rate limits
    import os
    token = os.getenv("GITHUB_TOKEN")
    
    finder = RepoFinder(github_token=token)
    repos = finder.discover_daily()
    saved_file = finder.save_repos(repos)
    print(f"Saved {len(repos)} repositories to {saved_file}") 