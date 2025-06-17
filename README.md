# Kinz AI Corporate Search Agent

A sophisticated corporate search system that uses vector embeddings and LLM-powered entity extraction to search and classify business entities, addresses, and person-related information.

## Features

- **Entity Extraction**: Automatically extracts key entities, addresses, business fields, and person information from natural language queries
- **Multi-level Classification**: Supports hierarchical classification (Sector → Subsector → Group → Filter)
- **Address Recognition**: Handles location-based searches across cities, districts, subdistricts, and streets
- **Business Field Filtering**: Supports employee range, business type, and registration date filtering
- **Person Search**: Enables searching by title groups, gender, and salutations
- **Vector Search**: Uses Pinecone for efficient similarity search with OpenAI embeddings

## Setup

### Prerequisites

- Python 3.8+
- OpenAI API key
- Pinecone API key

### Installation

1. Clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate virtual environment: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Copy `.env.example` to `.env` and add your API keys
6. Generate embeddings by running the generate_*.py scripts
7. Run the application: `python app.py`

## Usage

Send natural language queries to search for businesses and people. Examples:

- "banking companies in Amman"
- "IT companies with more than 50 employees"
- "Mr. John from top management"
- "companies registered after 2010 in Abdali"

## Security Notes

- Never commit API keys or sensitive data
- Keep your `.env` file local and secure
- Regularly rotate your API keys
