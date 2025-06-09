# Bayesian RL Agent - Learning What Works & What Doesn't

## What's This About?

This is a Bayesian reinforcement learning agent that reads new RL research, updates its beliefs about what techniques work/don't work, and reports findings daily. Think of it as an AI research assistant that gets smarter about RL over time.

## The Core Idea

Based on recent research (Bayesian Bellman Operators, Wasserstein Believer methods), we're building an agent that:

1. **Scrapes RL papers/repos** - Monitors ArXiv, GitHub, major conferences
2. **Updates beliefs** - Uses Bayesian methods to track which techniques are promising vs overhyped  
3. **Reports daily** - Summarizes interesting findings and belief updates
4. **Gets smarter** - Learns from tracking prediction accuracy over time

## Current Implementation

### What's Working:
- 📁 Repository structure with modular components
- 🔍 Basic paper/repo discovery pipeline  
- 📊 Bayesian belief tracking for technique effectiveness
- 📝 Daily report generation

### What's Next:
- [ ] Better paper parsing (extract actual results, not just abstracts)
- [ ] GitHub repo analysis (star growth, commit patterns, real usage)
- [ ] More sophisticated belief models (maybe try Bayesian neural networks?)
- [ ] Web interface for browsing beliefs and reports
- [ ] Integration with major RL conferences/workshops
- [ ] Track prediction accuracy and adapt accordingly

## Technical Approach

We're using insights from:
- **Bayesian Bellman Operators** - Uncertainty over operators rather than Q-functions
- **Wasserstein Believer** - Learning belief updates with theoretical guarantees
- **HEBO library** - Practical Bayesian optimization tools

The agent maintains probabilistic beliefs about technique effectiveness and updates them as new evidence comes in. No hype, just math.

## Quick Start

```bash
# Setup environment
pip install -r requirements.txt

# Run daily discovery and belief update
python -m src.agent.daily_run

# Check current beliefs
python -m src.analysis.view_beliefs

# Generate report
python -m src.reporting.daily_report
```

## Folder Structure

```
├── src/
│   ├── discovery/     # Paper/repo finding
│   ├── analysis/      # Belief updating logic  
│   ├── agent/         # Main agent coordination
│   └── reporting/     # Daily summaries
├── data/
│   ├── papers/        # Discovered papers
│   ├── repos/         # Repo analysis
│   └── beliefs/       # Belief state history
└── reports/           # Daily outputs
```

## Philosophy

No fancy marketing - just a tool that helps track what's actually working in RL research. Gets things wrong sometimes, but learns from mistakes. Perfect for researchers who want signal over noise. 