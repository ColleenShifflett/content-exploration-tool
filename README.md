# Content Exploration Tool

An AI-powered content analysis tool that helps you transform web content into a searchable, intelligent inventory with advanced insights and social media generation capabilities.

## Features 
View screenshot demos here: https://www.canva.com/design/DAGoSwrzpGs/cslVBGqfsA83QU5En40paw/view?utm_content=DAGoSwrzpGs&utm_campaign=designshare&utm_medium=link2&utm_source=uniquelinks&utlId=h66d7c45e0b

### Content Analysis
- **Website Crawling**: Extract content from entire websites
- **Content Inventory**: Build a searchable database of all your content
- **Semantic Search**: Find content by meaning, not just keywords

### AI-Powered Insights
- **Theme Analysis**: Identify the main topics and patterns in your content
- **Content Gaps**: Discover what topics you're missing
- **Quality Assessment**: Evaluate content effectiveness

### Interactive Chat
- **RAG-Powered Conversations**: Chat with your content using LangChain's RAG
- **Ask Questions**: "What topics do we cover most?" or "What's our stance on AI?"
- **Source Citations**: See exactly which content the answers come from

### Social Media Content
- **Style-Matched Generation**: Create social posts matching your content's voice
- **Multi-Platform Support**: Twitter, LinkedIn, Instagram content variations
- **Content Repurposing**: Turn articles into thread ideas, social posts, and more

## Technology Stack

- **LangChain**: For agents, RAG, and content processing
- **OpenAI**: For embeddings and language models
- **Streamlit**: User interface and visualization
- **ChromaDB**: Vector database for semantic search
- **BeautifulSoup**: Web scraping and content extraction

## Getting Started

### Prerequisites
- Python 3.9+
- OpenAI API key

### Installation

1. Clone the repository
```bash
git clone https://github.com/ColleenShifflett/content-exploration-tool.git
cd content-exploration-tool
```

2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Set up your OpenAI API key
```bash
# Create a .env file
cp .env.example .env
# Edit .env and add your API key
```

5. Run the application
```bash
streamlit run main.py
```

## Usage Guide

### Adding Content
- **Single URLs**: Add individual pieces of content
- **Website Crawling**: Analyze entire websites (up to 20 pages)
- **Text Input**: Directly paste content for analysis

### Analyzing Content
- **AI-Powered Analysis**: Get insights about themes, patterns, and gaps
- **Content Search**: Find specific information in your content
- **Visual Reports**: View charts and statistics about your content library

### Generating Social Media
- **Platform Selection**: Choose Twitter, LinkedIn, Instagram, or thread formats
- **Style Matching**: Posts match your content's voice and tone
- **Bulk Generation**: Create multiple posts from different content pieces

## Implementation Details

### RAG Architecture
This tool implements Retrieval Augmented Generation (RAG) using LangChain's components:
- Text chunking and embedding
- Vector storage in ChromaDB
- Query processing and response generation

### Agent System
Custom ReAct agents analyze content using specialized tools:
- Content theme analysis
- Content type detection
- Quality assessment
- Gap identification

### Social Media Generation
ML-powered content repurposing with:
- Style analysis
- Platform-specific formatting
- Brand voice preservation

## License

MIT License

## Acknowledgments

- Built with LangChain
- UI powered by Streamlit
- Special thanks to the open-source AI community
