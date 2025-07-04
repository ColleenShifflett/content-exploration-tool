import logging
from langchain.agents import initialize_agent, AgentType, Tool
from langchain_openai import ChatOpenAI
from langchain.schema import Document
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from typing import List, Dict
import pandas as pd
import config

# Set up logger
logger = logging.getLogger(__name__)

class ContentAnalysisAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model_name=config.MODEL_NAME,
            openai_api_key=config.OPENAI_API_KEY,
            temperature=0.3
        )
        
        # Store content for tools to access
        self.current_content = []
        
        # Create analysis tools that work with actual content
        self.tools = [
            Tool(
                name="Content_Theme_Analyzer",
                func=self._analyze_themes,
                description="Analyzes the actual themes and topics in the content library"
            ),
            Tool(
                name="Content_Type_Analyzer", 
                func=self._analyze_content_types,
                description="Examines the types and formats of content in the library"
            ),
            Tool(
                name="Content_Quality_Analyzer",
                func=self._analyze_content_quality,
                description="Assesses the quality and depth of content in the library"
            ),
            Tool(
                name="Content_Gap_Analyzer",
                func=self._analyze_content_gaps,
                description="Identifies gaps and opportunities in the content library"
            )
        ]
        
        # Initialize LangChain agent with better configuration
        self.agent = initialize_agent(
            self.tools,
            self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=False,
            max_iterations=6,
            max_execution_time=90,
            early_stopping_method="generate"
        )
        
        # Create specialized chains
        self._setup_analysis_chains()
    
    def _setup_analysis_chains(self):
        """Setup specialized LangChain chains for different analyses"""
        
        # Comprehensive analysis chain that uses actual content
        self.comprehensive_prompt = PromptTemplate(
            input_variables=["content_details", "content_stats"],
            template="""You are analyzing a specific content library. Provide detailed, personalized insights based on this ACTUAL content:

            CONTENT DETAILS:
            {content_details}

            LIBRARY STATISTICS:
            {content_stats}

            Provide a comprehensive analysis covering:

            1. CONTENT THEMES: What specific themes and topics appear in THIS library? Quote actual titles/summaries.

            2. CONTENT CHARACTERISTICS: What types of content dominate? What's the style/approach?

            3. STRENGTHS: What does THIS specific library do well based on the actual content?

            4. CONTENT GAPS: What topics or types are notably absent from THIS library?

            5. STRATEGIC RECOMMENDATIONS: Specific, actionable recommendations based on THIS content.

            Be specific and reference actual content from the library. Avoid generic advice.

            ANALYSIS:"""
        )
        self.comprehensive_chain = LLMChain(llm=self.llm, prompt=self.comprehensive_prompt)
        
        # Trend analysis chain
        trend_prompt = PromptTemplate(
            input_variables=["content_summaries"],
            template="""Analyze the following actual content summaries to identify patterns and trends:

            {content_summaries}

            Identify:
            1. Common themes that appear across multiple pieces
            2. Content style and approach patterns
            3. Topic evolution or focus areas
            4. Audience and purpose patterns

            Base your analysis only on the actual content provided above.

            TRENDS:"""
        )
        self.trend_chain = LLMChain(llm=self.llm, prompt=trend_prompt)
        
        # Strategy chain
        strategy_prompt = PromptTemplate(
            input_variables=["content_analysis"],
            template="""Based on this SPECIFIC content analysis, provide targeted strategic recommendations:

            {content_analysis}

            Provide specific recommendations for:
            1. Content types to prioritize next
            2. Topics to explore or expand
            3. Content gaps to fill
            4. Ways to improve content organization
            5. Opportunities for content repurposing

            Make recommendations specific to the analyzed content, not generic advice.

            STRATEGY:"""
        )
        self.strategy_chain = LLMChain(llm=self.llm, prompt=strategy_prompt)
    
    def _analyze_themes(self, content_items):
        """Analyze themes from the current content."""
        try:
            # Get current content summaries
            content_samples = []
            for item in content_items:
                if item.summary:
                    content_samples.append(f"Title: {item.title}\nSummary: {item.summary}")
            
            if not content_samples:
                return "No content available for theme analysis."
            
            # Construct prompt for theme analysis
            theme_prompt = PromptTemplate(
                input_variables=["content_samples"],
                template="""
                Analyze the following content samples to identify key themes and patterns:

                {content_samples}

                Identify:
                1. Main themes and topics
                2. Common patterns in content style
                3. Target audience indicators
                4. Content purpose and goals

                THEME ANALYSIS:"""
            )
            
            # Create and run the chain
            theme_chain = LLMChain(llm=self.llm, prompt=theme_prompt)
            return theme_chain.run(content_samples="\n\n".join(content_samples))
            
        except Exception as e:
            logger.error(f"Error in theme analysis: {str(e)}")
            return f"Error analyzing themes: {str(e)}"
    
    def _analyze_content_types(self, input_text: str) -> str:
        """Tool function for content type analysis"""
        try:
            if not self.current_content:
                return "No content loaded for analysis"
            
            # Analyze actual content types and characteristics
            type_analysis = {}
            word_counts = []
            
            for item in self.current_content:
                content_type = item.get('content_type', 'unknown')
                word_count = item.get('word_count', 0)
                
                type_analysis[content_type] = type_analysis.get(content_type, 0) + 1
                word_counts.append(word_count)
            
            avg_words = sum(word_counts) / len(word_counts) if word_counts else 0
            
            # Get sample titles for each type
            type_samples = {}
            for item in self.current_content[:10]:
                content_type = item.get('content_type', 'unknown')
                if content_type not in type_samples:
                    type_samples[content_type] = []
                if len(type_samples[content_type]) < 3:
                    type_samples[content_type].append(item.get('title', 'Untitled'))
            
            result = f"""Content Type Analysis:
            - Types found: {dict(type_analysis)}
            - Average word count: {avg_words:.0f}
            - Sample titles by type: {type_samples}
            
            The library contains {len(self.current_content)} items with these characteristics."""
            
            return result
        except Exception as e:
            return f"Error analyzing content types: {str(e)}"
    
    def _analyze_content_quality(self, input_text: str) -> str:
        """Tool function for content quality analysis"""
        try:
            if not self.current_content:
                return "No content loaded for analysis"
            
            # Analyze quality indicators
            word_counts = [item.get('word_count', 0) for item in self.current_content]
            titles_with_summaries = sum(1 for item in self.current_content if item.get('summary') and item.get('summary') != 'No summary')
            
            quality_indicators = {
                'total_items': len(self.current_content),
                'items_with_summaries': titles_with_summaries,
                'avg_word_count': sum(word_counts) / len(word_counts) if word_counts else 0,
                'shortest_content': min(word_counts) if word_counts else 0,
                'longest_content': max(word_counts) if word_counts else 0
            }
            
            # Sample some actual summaries
            sample_summaries = []
            for item in self.current_content[:3]:
                if item.get('summary') and item.get('summary') != 'No summary':
                    sample_summaries.append(f"'{item.get('title')}': {item.get('summary')[:150]}...")
            
            # Join summaries with newlines
            sample_text = '\n'.join(sample_summaries[:3])
            
            result = f"""Content Quality Assessment:
            - Total pieces: {quality_indicators['total_items']}
            - Items with AI summaries: {quality_indicators['items_with_summaries']}
            - Average length: {quality_indicators['avg_word_count']:.0f} words
            - Length range: {quality_indicators['shortest_content']}-{quality_indicators['longest_content']} words
            
            Sample content quality:
            {sample_text}"""
            
            return result
        except Exception as e:
            return f"Error analyzing content quality: {str(e)}"
    
    def _analyze_content_gaps(self, input_text: str) -> str:
        """Tool function for content gap analysis"""
        try:
            if not self.current_content:
                return "No content loaded for analysis"
            
            # Extract topics from titles and summaries
            all_text = ""
            for item in self.current_content:
                title = item.get('title', '')
                summary = item.get('summary', '')
                all_text += f"{title} {summary} "
            
            gap_prompt = f"""
            Based on this content library's actual topics and themes:
            
            {all_text[:2000]}
            
            Identify potential content gaps and opportunities. What topics, formats, or approaches might be missing?
            Be specific about gaps you see in THIS particular collection."""
            
            result = self.llm.predict(gap_prompt)
            return result
        except Exception as e:
            return f"Error analyzing content gaps: {str(e)}"
    
    def analyze_content_library(self, content_items: List[Dict]) -> Dict:
        """Perform comprehensive analysis using actual content data"""
        try:
            # Store content for tools to access
            self.current_content = content_items
            
            # Prepare detailed content information
            content_details = ""
            for i, item in enumerate(content_items[:8]):  # Limit to prevent token overflow
                title = item.get('title', 'Untitled')
                summary = item.get('summary', 'No summary')[:200]
                word_count = item.get('word_count', 0)
                url = item.get('url', 'No URL')
                
                content_details += f"""
                {i+1}. Title: {title}
                   Summary: {summary}
                   Word Count: {word_count}
                   Source: {url}
                   ---"""
            
            # Prepare statistics
            total_items = len(content_items)
            total_words = sum(item.get('word_count', 0) for item in content_items)
            content_types = {}
            for item in content_items:
                ctype = item.get('content_type', 'unknown')
                content_types[ctype] = content_types.get(ctype, 0) + 1
            
            content_stats = f"""
            - Total items: {total_items}
            - Total words: {total_words:,}
            - Average words per item: {total_words/total_items:.0f}
            - Content types: {content_types}
            """
            
            # Use the agent to analyze with actual content
            try:
                analysis_query = f"""
                Analyze this content library using the available tools. Focus on:
                1. What themes appear in the content?
                2. What types of content dominate?
                3. What's the quality and depth like?
                4. What gaps exist?
                
                Provide specific insights based on the actual content, not generic advice."""
                
                agent_result = self.agent.run(analysis_query)
            except Exception as agent_error:
                print(f"Agent failed: {agent_error}")
                # Fallback to direct comprehensive analysis
                agent_result = self.comprehensive_chain.run(
                    content_details=content_details,
                    content_stats=content_stats
                )
            
            # Run specialized analyses
            content_summaries = "\n".join([
                f"{item.get('title', 'Untitled')}: {item.get('summary', 'No summary')[:150]}"
                for item in content_items[:8]
            ])
            
            trend_analysis = self.trend_chain.run(content_summaries=content_summaries)
            strategy_recommendations = self.strategy_chain.run(content_analysis=trend_analysis)
            
            return {
                "agent_analysis": agent_result,
                "trend_analysis": trend_analysis,
                "strategy_recommendations": strategy_recommendations,
                "success": True
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "success": False
            }
    
    def generate_content_tags(self, content_items: List[Dict]) -> Dict:
        """Generate intelligent tags for content using LangChain"""
        try:
            tag_results = {}
            
            for item in content_items:
                content_text = f"{item.get('title', '')} {item.get('summary', '')}"
                
                tag_prompt = f"""
                Generate 3-5 relevant tags for this specific content:
                
                Title: {item.get('title', 'Untitled')}
                Summary: {item.get('summary', 'No summary')[:300]}
                
                Tags should be:
                - Specific to this content
                - Useful for categorization
                - Single words or short phrases
                
                Tags (comma-separated):"""
                
                try:
                    tags = self.llm.predict(tag_prompt)
                    tag_results[item.get('id', 'unknown')] = [tag.strip() for tag in tags.split(',')]
                except Exception as tag_error:
                    print(f"Error generating tags for {item.get('title', 'unknown')}: {tag_error}")
                    tag_results[item.get('id', 'unknown')] = ["analysis-error"]
            
            return {
                "tags": tag_results,
                "success": True
            }
        except Exception as e:
            return {
                "error": str(e),
                "success": False
            }

    def generate_social_media_posts(self, content_items: List[Dict], platforms: List[str], num_posts: int) -> Dict:
        """Generate social media posts for different platforms based on content library"""
        try:
            # Get current content summaries
            content_samples = []
            for item in content_items:
                if item.get('summary'):
                    content_samples.append(f"Title: {item.get('title', '')}\nSummary: {item.get('summary', '')}")
            
            if not content_samples:
                return {
                    "error": "No content available for social media post generation.",
                    "success": False
                }
            
            # Analyze writing style
            style_prompt = PromptTemplate(
                input_variables=["content_samples"],
                template="Analyze the writing style of these content pieces:\n\n{content_samples}\n\nIdentify:\n1. Tone and voice\n2. Key messaging points\n3. Target audience\n4. Brand voice characteristics\n\nSTYLE ANALYSIS:"
            )
            
            style_chain = LLMChain(llm=self.llm, prompt=style_prompt)
            style_analysis = style_chain.run(content_samples="\n\n".join(content_samples))
            
            # Prepare statistics
            content_stats = f"Total Items: {len(content_items)}\nItems with Summaries: {len(content_samples)}\nAverage Word Count: {sum(len(item.get('summary', '').split()) for item in content_items if item.get('summary')) / len(content_samples) if content_samples else 0:.1f}\nLength Range: {min(len(item.get('summary', '').split()) for item in content_items if item.get('summary')) if content_samples else 0} - {max(len(item.get('summary', '').split()) for item in content_items if item.get('summary')) if content_samples else 0} words"
            
            # Generate platform-specific posts
            platform_styles = {
                "LinkedIn": "professional, thought leadership, industry insights",
                "Twitter": "concise, engaging, with relevant hashtags",
                "Facebook": "conversational, community-focused, with call-to-action",
                "Instagram": "visual, storytelling, with relevant hashtags"
            }
            
            social_posts = {"style_analysis": style_analysis}
            
            for platform in platforms:
                if platform not in platform_styles:
                    continue
                    
                style = platform_styles[platform]
                platform_posts = []
                
                # Select content items to generate posts from
                selected_items = content_items[:num_posts]
                
                for item in selected_items:
                    post_prompt = PromptTemplate(
                        input_variables=["style_analysis", "content_stats", "platform", "style", "title", "summary"],
                        template="Based on this content analysis and style:\n\nStyle Analysis:\n{style_analysis}\n\nContent Statistics:\n{content_stats}\n\nCreate a {platform} post based on this content:\nTitle: {title}\nSummary: {summary}\n\nThe post should be {style}.\nInclude relevant hashtags where appropriate.\n\nPOST:"
                    )
                    
                    post_chain = LLMChain(llm=self.llm, prompt=post_prompt)
                    post = post_chain.run(
                        style_analysis=style_analysis,
                        content_stats=content_stats,
                        platform=platform,
                        style=style,
                        title=item.get('title', ''),
                        summary=item.get('summary', '')[:300]
                    )
                    
                    platform_posts.append({
                        'content_title': item.get('title', ''),
                        'post': post,
                        'source_url': item.get('url', '#')
                    })
                
                social_posts[platform] = platform_posts
            
            return {
                "social_posts": social_posts,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error generating social media posts: {str(e)}")
            return {
                "error": str(e),
                "success": False
            }
    
    def generate_content_marketing_ideas(self, content_items):
        """Generate creative marketing ideas based on the content library."""
        try:
            # Get current content summaries
            content_samples = []
            for item in content_items:
                if item.get('summary'):
                    content_samples.append(f"Title: {item.get('title', '')}\nSummary: {item.get('summary', '')}")
            
            if not content_samples:
                return {
                    "success": False,
                    "error": "No content available for marketing idea generation."
                }
            
            # Construct prompt for marketing ideas
            ideas_prompt = PromptTemplate(
                input_variables=["content_samples"],
                template="""
                Based on these content pieces, generate creative marketing ideas:

                {content_samples}

                Generate 5 innovative marketing ideas that:
                1. Leverage existing content effectively
                2. Target the identified audience
                3. Align with the content's style and tone
                4. Include specific implementation suggestions
                5. Consider different marketing channels

                MARKETING IDEAS:"""
            )
            
            # Create and run the chain
            ideas_chain = LLMChain(llm=self.llm, prompt=ideas_prompt)
            marketing_ideas = ideas_chain.run(content_samples="\n\n".join(content_samples))
            
            return {
                "success": True,
                "marketing_ideas": marketing_ideas
            }
            
        except Exception as e:
            logger.error(f"Error generating marketing ideas: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
