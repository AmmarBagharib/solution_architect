# Cloud Architecture Recommendation System
**Project Directory Structure**
``` 
solution_architect/
├── README.md
├── requirements.txt
├── app.py
├── run.sh
├── config/
│   ├── __init__.py
│   ├── settings.py # Application settings and configuration
│   └── cloud_providers.py # Cloud provider information and endpoints
├── agents/
│   ├── __init__.py
│   ├── base_agent.py # Base agent with common functionality
│   ├── master_agent.py # Orchestrator agent that coordinates other agents
│   ├── scraping_agent.py # Agent for scraping cloud pricing data
│   ├── solution_architect_agent.py # Agent for architecture recommendations
│   └── mathematics_agent.py # Agent for cost calculations and validation
├── utils/
│   ├── __init__.py
│   ├── data_processing.py # Data transformation utilities
│   ├── cost_calculations.py # Helper functions for cost calculations
│   └── validation.py # Input validation utilities
├── data/
│   ├── __init__.py
│   ├── cache/ # Cache directory for scraped pricing data
│   │   └── .gitkeep
│   ├── schemas/ # JSON schemas for validation
│   │   └── requirements_schema.json
│   └── templates/  # Architecture templates and patterns
│       └── architecture_patterns.json
├── static/
│   ├── css/ # Streamlit custom CSS
│   │   └── style.css
│   └── images/ # Images for the web application
│       └── logo.png
└── tests/
    ├── __init__.py
    ├── test_master_agent.py
    ├── test_scraping_agent.py
    ├── test_solution_architect_agent.py
    └── test_mathematics_agent.py
```

## Overview
This project implements a multi-agent framework that helps users determine the most cost-effective and appropriate cloud architecture based on their specific requirements. The system scrapes pricing information from major cloud providers, analyzes user requirements, and generates architecture recommendations while ensuring cost constraints are met.

## Key Features
- Interactive user requirement gathering
- Automated web scraping of cloud provider pricing
- Cost-effective architecture recommendations
- Budget constraint validation
- Comprehensive reasoning for recommendations

## Architecture
The system consists of four main agents:

1. **Master Agent**: Interfaces with users, collects requirements, and orchestrates the workflow between other agents
2. **Scraping Agent**: Collects up-to-date pricing information from cloud providers using crawl4ai
3. **Solution Architect Agent**: Analyzes requirements and develops appropriate architecture recommendations
4. **Mathematics Agent**: Validates that recommended architectures meet budget constraints and calculates TCO

## Technical Requirements
- Python 3.8+
- Open-source models and tools only
- No paid services or APIs

## Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/cloud-architecture-recommendation.git
cd cloud-architecture-recommendation

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## API Setup
1. Copy `.env.template` to `.env`:
```bash
cp .env.template .env
```

2. Add your DeepSeek API key to the `.env` file:
```
DEEPSEEK_API_KEY=your_api_key_here
```

## Usage
```bash
# Run the Streamlit web application
streamlit run app.py
```

This will launch a web interface where you can:
1. Input your cloud architecture requirements
2. Generate architecture recommendations
3. View detailed cost analysis
4. Generate comprehensive reports
5. Export reports as JSON

## Current Status
- Development phase in progress
- Implemented Scraping Agent with caching mechanism
- Implemented Mathematics Agent for cost calculations and budget validation
- Implemented Solution Architect Agent with architecture patterns
- Working on Master Agent implementation