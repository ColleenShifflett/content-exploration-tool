import streamlit as st
import pandas as pd
import os
from content_processor import ContentProcessor
from database import ContentDatabase
from site_crawler import SiteCrawler
from rag_chat import RAGChatSystem
from content_analysis import ContentAnalysisAgent
from urllib.parse import urlparse
import config

# Initialize components
@st.cache_resource
def initialize_app():
    # Create data directory if it doesn't exist
    os.makedirs("./data", exist_ok=True)
    
    processor = ContentProcessor()
    database = ContentDatabase(config.SQLITE_DB_PATH, config.CHROMA_DB_PATH)
    rag_system = RAGChatSystem(config.CHROMA_DB_PATH)
    analysis_agent = ContentAnalysisAgent()
    
    return processor, database, rag_system, analysis_agent

def main():
    st.set_page_config(page_title="Content Inventory MVP", page_icon="📚", layout="wide")
    
    st.title("📚 Personal Web Content Librarian")
    st.markdown("Transform your web content into a searchable, intelligent inventory with AI-powered analysis")
    
    # Check for OpenAI API key
    if not config.OPENAI_API_KEY:
        st.error("⚠️ OpenAI API key not found. Please add OPENAI_API_KEY to your environment variables.")
        return
    
    try:
        processor, database, rag_system, analysis_agent = initialize_app()
    except Exception as e:
        st.error(f"Error initializing app: {str(e)}")
        return
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    mode = st.sidebar.radio("Choose an action:", [
        "Add Content", 
        "Crawl Entire Site", 
        "Chat with Content", 
        "Content Analysis",
        "Search Content", 
        "View Library",
        "Export Data"
    ])
    
    if mode == "Add Content":
        st.header("📥 Add New Content")
        
        # Content input options
        input_type = st.radio("Content source:", ["URL", "Text"])
        
        if input_type == "URL":
            url = st.text_input("Enter URL:", placeholder="https://example.com/article")
            if st.button("Process URL") and url:
                with st.spinner("Processing content..."):
                    result = processor.process_content(url, "url")
                    
                    if result['success']:
                        # Store in database
                        database.store_content(
                            result['content_id'],
                            result['metadata'],
                            result['chunks'],
                            result['embeddings']
                        )
                        
                        st.success("✅ Content processed and stored successfully!")
                        st.write("**Title:**", result['metadata']['title'])
                        st.write("**Summary:**", result['metadata']['summary'])
                        st.write("**Word count:**", result['metadata']['word_count'])
                    else:
                        st.error(f"❌ Error processing content: {result['error']}")
        
        else:  # Text input
            title = st.text_input("Title (optional):", placeholder="My Document")
            text_content = st.text_area("Paste your text content:", height=300)
            
            if st.button("Process Text") and text_content:
                with st.spinner("Processing content..."):
                    # Add title to content if provided
                    if title:
                        full_content = f"{title}\n\n{text_content}"
                    else:
                        full_content = text_content
                    
                    result = processor.process_content(full_content, "text")
                    
                    if result['success']:
                        # Store in database
                        database.store_content(
                            result['content_id'],
                            result['metadata'],
                            result['chunks'],
                            result['embeddings']
                        )
                        
                        st.success("✅ Content processed and stored successfully!")
                        st.write("**Title:**", result['metadata']['title'])
                        st.write("**Summary:**", result['metadata']['summary'])
                        st.write("**Word count:**", result['metadata']['word_count'])
                    else:
                        st.error(f"❌ Error processing content: {result['error']}")
    
    elif mode == "Crawl Entire Site":
        st.header("🕷️ Crawl Entire Website")
        st.markdown("Extract content from all pages of a website and create a comprehensive inventory.")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            site_url = st.text_input("Enter website URL:", placeholder="https://example.com")
        
        with col2:
            max_pages = st.number_input("Max pages to crawl:", min_value=1, max_value=20, value=5)
            delay = st.number_input("Delay between requests (seconds):", min_value=0.5, max_value=5.0, value=1.0, step=0.5)
        
        # Options
        st.subheader("Processing Options")
        col1, col2 = st.columns(2)
        with col1:
            generate_summaries = st.checkbox("Generate AI summaries (slower but more detailed)", value=True)
        with col2:
            store_in_db = st.checkbox("Store in searchable database", value=True)
        
        if st.button("🚀 Start Crawling") and site_url:
            # Initialize crawler
            crawler = SiteCrawler(max_pages=max_pages, delay=delay)
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            results_container = st.empty()
            
            # Start crawling
            status_text.text("Starting crawl...")
            crawled_pages = crawler.crawl_site(site_url)
            
            if crawled_pages:
                status_text.text("Processing pages...")
                
                # Process each page
                processed_data = []
                for i, page_data in enumerate(crawled_pages):
                    progress = (i + 1) / len(crawled_pages)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing page {i+1}/{len(crawled_pages)}: {page_data['title'][:50]}...")
                    
                    if page_data['status'] == 'success' and page_data['content']:
                        # Generate summary if requested
                        if generate_summaries:
                            summary = processor.generate_summary(page_data['content'])
                        else:
                            summary = f"Content from {page_data['url']}"
                        
                        # Store in database if requested
                        if store_in_db:
                            result = processor.process_content(page_data['url'], "url")
                            if result['success']:
                                database.store_content(
                                    result['content_id'],
                                    result['metadata'],
                                    result['chunks'],
                                    result['embeddings']
                                )
                        
                        processed_data.append({
                            'URL': page_data['url'],
                            'Title': page_data['title'],
                            'Summary': summary,
                            'Word Count': page_data['word_count'],
                            'Status': page_data['status']
                        })
                    else:
                        processed_data.append({
                            'URL': page_data['url'],
                            'Title': page_data['title'],
                            'Summary': f"Failed to process: {page_data['status']}",
                            'Word Count': page_data['word_count'],
                            'Status': page_data['status']
                        })
                
                # Display results
                status_text.text("✅ Crawling complete!")
                progress_bar.progress(1.0)
                
                st.success(f"Successfully crawled {len(crawled_pages)} pages!")
                
                # Create and display table
                df = pd.DataFrame(processed_data)
                st.subheader("📊 Site Content Inventory")
                st.dataframe(df, use_container_width=True)
                
                # Download option
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv,
                    file_name=f"site_inventory_{urlparse(site_url).netloc}.csv",
                    mime="text/csv"
                )
                
                # Summary stats
                st.subheader("📈 Summary Statistics")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Pages", len(df))
                with col2:
                    st.metric("Successful Crawls", len(df[df['Status'] == 'success']))
                with col3:
                    st.metric("Total Word Count", df['Word Count'].sum())
                with col4:
                    st.metric("Average Words/Page", round(df['Word Count'].mean()) if len(df) > 0 else 0)
                
            else:
                st.error("No pages were successfully crawled. Please check the URL and try again.")
    
    elif mode == "Chat with Content":
        st.header("💬 Chat with Your Content")
        st.markdown("Ask questions about your content library using AI-powered retrieval")
        
        # Check if there's content in the database
        content_items = database.get_all_content()
        
        if not content_items:
            st.warning("⚠️ No content found in your library. Add some content first!")
            st.markdown("Use **'Add Content'** or **'Crawl Entire Site'** to build your content library.")
            return
        
        st.success(f"📚 Connected to your library with {len(content_items)} items")
        
        # Chat interface
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        
        # Display chat history
        for chat in st.session_state.chat_history:
            with st.chat_message(chat["role"]):
                st.write(chat["content"])
                if chat["role"] == "assistant" and "sources" in chat:
                    with st.expander("📎 Sources"):
                        for i, source in enumerate(chat["sources"]):
                            st.write(f"**Source {i+1}:** {source}")
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your content..."):
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.write(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    result = rag_system.chat(prompt)
                    
                    if result["success"]:
                        st.write(result["answer"])
                        
                        # Display sources if available
                        if result["source_documents"]:
                            sources = []
                            for doc in result["source_documents"]:
                                # Try to get source from metadata
                                if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                                    sources.append(doc.metadata['source'])
                                else:
                                    sources.append(doc.page_content[:100] + "...")
                            
                            if sources:
                                with st.expander("📎 Sources"):
                                    for i, source in enumerate(sources):
                                        st.write(f"**Source {i+1}:** {source}")
                                
                                # Add assistant message with sources to chat history
                                st.session_state.chat_history.append({
                                    "role": "assistant", 
                                    "content": result["answer"],
                                    "sources": sources
                                })
                            else:
                                # Add assistant message without sources to chat history
                                st.session_state.chat_history.append({
                                    "role": "assistant", 
                                    "content": result["answer"]
                                })
                        else:
                            # Add assistant message to chat history
                            st.session_state.chat_history.append({
                                "role": "assistant", 
                                "content": result["answer"]
                            })
                    else:
                        error_msg = result["answer"]
                        st.error(error_msg)
                        st.session_state.chat_history.append({
                            "role": "assistant", 
                            "content": error_msg
                        })
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            rag_system.clear_memory()
            st.experimental_rerun()
    
    elif mode == "Content Analysis":
        st.header("🔍 Content Analysis")
        st.markdown("Get AI-powered insights about your content library using LangChain agents")
        
        content_items = database.get_all_content()
        
        if not content_items:
            st.warning("⚠️ No content found in your library. Add some content first!")
            return
        
        st.success(f"📊 Analyzing {len(content_items)} items in your library")
        
        # Analysis options
        st.subheader("Analysis Options")
        analysis_type = st.radio("Choose analysis type:", [
            "Complete Library Analysis",
            "Generate Content Tags",
            "Custom Analysis"
        ])
        
        if analysis_type == "Complete Library Analysis":
            if st.button("🔬 Run Complete Analysis"):
                with st.spinner("Running comprehensive analysis using LangChain agents..."):
                    results = analysis_agent.analyze_content_library(content_items)
                    
                    if results["success"]:
                        # Display results in tabs
                        tab1, tab2, tab3 = st.tabs(["Agent Analysis", "Trend Analysis", "Strategy Recommendations"])
                        
                        with tab1:
                            st.subheader("🤖 LangChain Agent Analysis")
                            st.write(results["agent_analysis"])
                        
                        with tab2:
                            st.subheader("📈 Trend Analysis")
                            st.write(results["trend_analysis"])
                        
                        with tab3:
                            st.subheader("💡 Strategy Recommendations")
                            st.write(results["strategy_recommendations"])
                    else:
                        st.error(f"Analysis failed: {results['error']}")
        
        elif analysis_type == "Generate Content Tags":
            if st.button("🏷️ Generate Tags"):
                with st.spinner("Generating intelligent tags..."):
                    results = analysis_agent.generate_content_tags(content_items)
                    
                    if results["success"]:
                        st.subheader("🏷️ Generated Tags")
                        
                        # Display tags in a nice format
                        tag_data = []
                        for content_id, tags in results["tags"].items():
                            # Find the content item
                            content_item = next((item for item in content_items if item['id'] == content_id), None)
                            if content_item:
                                tag_data.append({
                                    'Title': content_item['title'],
                                    'Tags': ', '.join(tags),
                                    'URL': content_item.get('url', 'N/A')
                                })
                        
                        if tag_data:
                            df = pd.DataFrame(tag_data)
                            st.dataframe(df, use_container_width=True)
                            
                            # Download tags
                            csv = df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Tags as CSV",
                                data=csv,
                                file_name="content_tags.csv",
                                mime="text/csv"
                            )
                    else:
                        st.error(f"Tag generation failed: {results['error']}")
        
        else:  # Custom Analysis
            st.subheader("✨ Custom Analysis")
            custom_query = st.text_area(
                "Describe what you want to analyze:",
                placeholder="e.g., 'What are the main themes across my content?' or 'Find content gaps in my library'"
            )
            
            if st.button("🔍 Run Custom Analysis") and custom_query:
                with st.spinner("Running custom analysis..."):
                    try:
                        # Use the agent to answer the custom query
                        content_summaries = []
                        for item in content_items[:10]:  # Limit for performance
                            summary = f"Title: {item.get('title', 'Unknown')}\nSummary: {item.get('summary', 'No summary')}"
                            content_summaries.append(summary)
                        
                        combined_content = "\n\n---\n\n".join(content_summaries)
                        
                        analysis_prompt = f"""
                        Based on this content library, please answer the following question:
                        {custom_query}
                        
                        Content Library:
                        {combined_content}
                        """
                        
                        result = analysis_agent.agent.run(analysis_prompt)
                        
                        st.subheader("📋 Analysis Results")
                        st.write(result)
                        
                    except Exception as e:
                        st.error(f"Custom analysis failed: {str(e)}")
    
    elif mode == "Search Content":
        st.header("🔍 Search Your Content")
        
        query = st.text_input("What are you looking for?", placeholder="Enter your search query...")
        num_results = st.slider("Number of results:", min_value=1, max_value=10, value=5)
        
        if st.button("Search") and query:
            with st.spinner("Searching..."):
                results = database.search_content(query, num_results)
                
                if results['metadata']:
                    st.success(f"Found {len(results['metadata'])} relevant items")
                    
                    for i, item in enumerate(results['metadata']):
                        with st.expander(f"📄 {item[2]} (Distance: {results['distances'][i]:.3f})"):
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                st.write("**Summary:**", item[3])
                                if item[1]:  # URL exists
                                    st.write("**Source:**", item[1])
                                st.write("**Word Count:**", item[5])
                                st.write("**Created:**", item[6])
                            
                            with col2:
                                st.write("**Relevant excerpt:**")
                                st.write(f"_{results['chunks'][i][:200]}..._")
                else:
                    st.info("No results found. Try a different search query.")
    
    elif mode == "View Library":
        st.header("📚 Your Content Library")
        
        content_items = database.get_all_content()
        
        if content_items:
            st.write(f"**Total items:** {len(content_items)}")
            
            # Convert to DataFrame for better display
            df = pd.DataFrame(content_items)
            df = df[['title', 'url', 'summary', 'word_count', 'content_type', 'created_at']]
            df.columns = ['Title', 'URL', 'Summary', 'Word Count', 'Type', 'Created']
            
            st.dataframe(df, use_container_width=True)
            
            # Download option
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Library as CSV",
                data=csv,
                file_name="content_library.csv",
                mime="text/csv"
            )
        else:
            st.info("Your library is empty. Add some content to get started!")
    
    else:  # Export Data
        st.header("�� Export Your Data")
        
        content_items = database.get_all_content()
        
        if content_items:
            st.write(f"**Total items in library:** {len(content_items)}")
            
            # Export options
            export_format = st.radio("Export format:", ["CSV", "JSON"])
            
            if export_format == "CSV":
                df = pd.DataFrame(content_items)
                csv_data = df.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv_data,
                    file_name="content_inventory_export.csv",
                    mime="text/csv"
                )
                
                # Preview
                st.subheader("Preview:")
                st.dataframe(df.head(), use_container_width=True)
            
            else:  # JSON
                import json
                json_data = json.dumps(content_items, indent=2, default=str)
                
                st.download_button(
                    label="📥 Download as JSON",
                    data=json_data,
                    file_name="content_inventory_export.json",
                    mime="application/json"
                )
                
                # Preview
                st.subheader("Preview:")
                st.json(content_items[0] if content_items else {})
        else:
            st.info("No content to export. Add some content first!")

if __name__ == "__main__":
    main()
