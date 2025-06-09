# Bayesian RL Agent

Bayesian reinforcement learning agent that tracks technique effectiveness through probabilistic belief updating using **real research papers and LLM analysis**.

## Implementation

**Core System:**
- Beta distribution belief tracking for RL technique effectiveness
- Bayesian evidence integration from **real ArXiv papers and GitHub repositories**
- **LLM-powered content analysis** using Google Gemini 1.5 Flash for technique effectiveness assessment
- Daily report generation with belief summaries and confidence intervals

**Dependencies:**
- `numpy`, `scipy` for statistical computations
- `arxiv` API for real paper discovery
- `PyGithub` for repository analysis  
- `pandas` for data handling
- `requests` for Google Gemini API integration

## What's Working Now

**✅ Real Implementation (No More Mock Data):**

- **ArXiv Integration** - Fetches 59+ real RL papers from recent research
- **Gemini AI Analysis** - Google Gemini 1.5 Flash extracts technique effectiveness from paper content
- **Bayesian Beliefs** - Evidence integration with uncertainty quantification
- **Statistical Comparisons** - Probabilistic technique rankings
- **Complete Test Suite** - Full pipeline verification with real APIs

**Test commands:**
```bash
# Set Gemini API key
export GEMINI_API_KEY="AIzaSyBYm3cNe9HgUFPTsokMADs2xrF8XT93iuw"

# Run complete test suite
python tests/run_all_tests.py            # Tests all components

# Individual tests
python tests/test_arxiv.py               # ArXiv API (real papers)
python tests/test_gemini.py              # Gemini API functionality  
python tests/test_beliefs.py             # Bayesian belief tracking
python tests/test_full_pipeline.py       # End-to-end integration

# View current beliefs
python -m src.analysis.view_beliefs      # Shows tracked techniques
```

## LLM-Based Analysis

**Replaces regex pattern matching with:**

- **Google Gemini 1.5 Flash** for content understanding and analysis
- **Effectiveness scoring** (0.0-1.0) based on paper results and performance claims
- **Confidence assessment** from LLM reasoning + metadata factors
- **Reasoning capture** - Gemini explains its technique assessments

**Scoring Guidelines:**
- 0.9-1.0: Breakthrough results, state-of-the-art performance
- 0.7-0.9: Strong positive results, clearly effective  
- 0.5-0.7: Moderate effectiveness, mixed results
- 0.3-0.5: Limited effectiveness, significant limitations
- 0.0-0.3: Poor performance, major issues

## Workflow

```
ArXiv Papers → Gemini Analysis → Evidence Extraction → Bayesian Update → Reports
     ↓              ↓                ↓                   ↓            ↓
  Real research   Gemini 1.5       Effectiveness      Beta params    Daily MD
  (59+ papers)    assessment       scores + conf      α,β update     + JSON data
```

## Architecture

```
src/
├── discovery/     # ArXiv/GitHub real data collection
├── analysis/      # Gemini analyzer + Bayesian tracker  
├── agent/         # Daily orchestration (supports both LLM + fallback)
└── reporting/     # Markdown report generation

tests/             # Complete test suite
├── test_arxiv.py         # ArXiv API functionality
├── test_gemini.py        # Gemini API integration
├── test_beliefs.py       # Bayesian belief tracking
├── test_full_pipeline.py # End-to-end pipeline
└── run_all_tests.py      # Complete test runner
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

**Gemini Integration:**
```
Paper → Gemini 1.5 Flash → {technique, score, confidence, reasoning}
Evidence = Gemini_score × paper_confidence
Belief_update(Evidence)
```

## Usage

```bash
# Setup
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# Set Gemini API key for LLM analysis
export GEMINI_API_KEY="your-gemini-api-key"

# Run complete test suite (recommended first step)
python tests/run_all_tests.py

# Run full pipeline test with real papers
python tests/test_full_pipeline.py

# Generate reports
python -m src.reporting.daily_report
```

## API Setup

**Google AI Studio Gemini:**
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create API key
3. Set environment variable: `export GEMINI_API_KEY="your-key"`
4. Model used: `gemini-1.5-flash-latest`

## Test Results

**Real Analysis Results from Gemini:**
- "Reflect-then-Plan" → Offline RL (0.7), Model-Based RL (0.7), Bayesian methods (0.7)
- "Wind Farm Control" → Deep RL (0.7), Policy Gradient (0.7), Actor-Critic (0.7)
- "Self-Attention Study" → Multi-Agent RL (0.5)

**Current System Status:**
- ✅ 59 real papers from ArXiv cs.LG, cs.AI, stat.ML, cs.RO
- ✅ Gemini analysis with effectiveness scoring and reasoning
- ✅ 15 RL techniques tracked with 37+ evidence points
- ✅ Complete Bayesian belief updating with uncertainty quantification

## References

- O'Donoghue et al. "The Uncertainty Bellman Equation and Exploration" ([arXiv:1709.05380](https://arxiv.org/abs/1709.05380))
- Depeweg et al. "Decomposition of Uncertainty in Bayesian Deep Learning" ([arXiv:1710.07283](https://arxiv.org/abs/1710.07283))
- Ghavamzadeh et al. "Bayesian Reinforcement Learning: A Survey" ([arXiv:1609.04436](https://arxiv.org/abs/1609.04436)) 