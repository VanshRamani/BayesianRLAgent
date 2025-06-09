# Bayesian RL Agent

Bayesian reinforcement learning agent that tracks technique effectiveness through probabilistic belief updating using **real research papers and LLM analysis**.

## Implementation

**Core System:**
- Beta distribution belief tracking for RL technique effectiveness
- Bayesian evidence integration from **real ArXiv papers and GitHub repositories**
- **LLM-powered content analysis** using OpenAI GPT-4o-mini for technique effectiveness assessment
- Daily report generation with belief summaries and confidence intervals

**Dependencies:**
- `numpy`, `scipy` for statistical computations
- `arxiv` API for real paper discovery
- `PyGithub` for repository analysis  
- `pandas` for data handling
- `requests` for OpenAI API integration

## What's Working Now

**✅ Real Implementation (No More Mock Data):**

- **ArXiv Integration** - Fetches 50+ real RL papers from recent research
- **LLM Analysis** - GPT-4o-mini extracts technique effectiveness from paper content
- **Bayesian Beliefs** - Evidence integration with uncertainty quantification
- **Statistical Comparisons** - Probabilistic technique rankings
- **Data Persistence** - Belief state and paper data saving

**Test commands:**
```bash
# Test ArXiv API (gets real papers)
python test_arxiv.py                     # Shows 3 real papers from ArXiv

# Run with real papers + LLM analysis (requires OpenAI API key)
export OPENAI_API_KEY="your-key-here"
python real_demo.py                      # Analyzes real papers with LLM

# Fallback demo with mock data (no API key needed)
python simple_demo.py                    # Uses hardcoded evidence

# View current beliefs
python -m src.analysis.view_beliefs      # Shows tracked techniques
```

## LLM-Based Analysis

**Replaces regex pattern matching with:**

- **OpenAI GPT-4o-mini** for content understanding
- **Effectiveness scoring** (0.0-1.0) based on paper results
- **Confidence assessment** from LLM + metadata factors
- **Reasoning capture** - LLM explains its assessments

**Scoring Guidelines:**
- 0.9-1.0: Breakthrough results, SOTA performance
- 0.7-0.9: Strong positive results, clearly effective  
- 0.5-0.7: Moderate effectiveness, mixed results
- 0.3-0.5: Limited effectiveness, significant limitations
- 0.0-0.3: Poor performance, major issues

## Workflow

```
ArXiv Papers → LLM Analysis → Evidence Extraction → Bayesian Update → Reports
     ↓              ↓                ↓                   ↓            ↓
  Real research   GPT-4o-mini    Effectiveness      Beta params    Daily MD
  (50+ papers)    assessment     scores + conf      α,β update     + JSON data
```

## Architecture

```
src/
├── discovery/     # ArXiv/GitHub real data collection
├── analysis/      # LLM analyzer + Bayesian tracker  
├── agent/         # Daily orchestration (supports both LLM + fallback)
└── reporting/     # Markdown report generation
```

## Technical Approach

Based on:
- [Bayesian Bellman Operators](https://arxiv.org/abs/2106.00426) - Uncertainty quantification in RL
- [Deep Bayesian RL](https://arxiv.org/abs/1802.04412) - Posterior inference over Q-functions  
- [Wasserstein Auto-Encoders](https://arxiv.org/abs/1711.01558) - Latent space belief updates

**Belief Updates:**
```
α_new = α + evidence_weight × positive_signal
β_new = β + evidence_weight × negative_signal
effectiveness = α/(α+β)
uncertainty = sqrt(αβ/((α+β)²(α+β+1)))
```

**LLM Integration:**
```
Paper → GPT-4o-mini → {technique, score, confidence, reasoning}
Evidence = LLM_score × paper_confidence
Belief_update(Evidence)
```

## Usage

```bash
# Setup
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install numpy scipy pandas arxiv PyGithub requests

# Set OpenAI API key for LLM analysis
export OPENAI_API_KEY="your-openai-api-key"

# Run real paper analysis (recommended)
python real_demo.py

# Run daily agent with LLM
python -m src.agent.daily_run --openai-api-key="your-key"

# Generate reports
python -m src.reporting.daily_report
```

## References

- O'Donoghue et al. "The Uncertainty Bellman Equation and Exploration" ([arXiv:1709.05380](https://arxiv.org/abs/1709.05380))
- Depeweg et al. "Decomposition of Uncertainty in Bayesian Deep Learning" ([arXiv:1710.07283](https://arxiv.org/abs/1710.07283))
- Ghavamzadeh et al. "Bayesian Reinforcement Learning: A Survey" ([arXiv:1609.04436](https://arxiv.org/abs/1609.04436)) 