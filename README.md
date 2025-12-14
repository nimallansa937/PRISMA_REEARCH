# Academic Research Agent

A domain-agnostic academic research agent that can search, verify, and synthesize academic papers from multiple sources.

## Features

- **Multi-Source Search**: Semantic Scholar, arXiv, PubMed, CrossRef
- **Domain Detection**: Automatically identifies research field (CS, Medicine, Biology, etc.)
- **Query Decomposition**: Generates multiple search variations with synonyms
- **Quality Scoring**: Scores papers based on citations, venue, recency, DOI
- **Iterative Refinement**: Automatically broadens search if insufficient results

## Installation

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

## Configuration

Copy `.env.example` to `.env` and add your API keys:

```bash
cp .env.example .env
```

## Usage

```python
from agent import ResearchAgent

agent = ResearchAgent()
results = agent.search("Mechanisms of cryptocurrency liquidation cascades")
print(results)
```

## Project Structure

```
academic-research-agent/
├── config/          # Configuration and settings
├── core/            # Query analysis and search strategy
├── sources/         # API clients for academic databases
├── validation/      # Quality control and verification
├── synthesis/       # Report generation
├── utils/           # Utilities (caching, rate limiting)
└── agent.py         # Main orchestrator
```

## API Endpoint (for React integration)

Run the FastAPI server:

```bash
uvicorn api:app --reload --port 8000
```

The React app can then call `http://localhost:8000/search?query=...`
