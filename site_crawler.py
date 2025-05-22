import requests
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import time
from typing import Set, List, Dict
import config
import streamlit as st

class SiteCrawler:
    def __init__(self, max_pages: int = 5, delay: float = 1.0):
        self.max_pages = max_pages
        self.delay = delay  # Delay between requests to be respectful
        self.visited_urls: Set[str] = set()
        self.found_urls: Set[str] = set()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0',
        }
        self.debug_info = []
    
    def is_valid_url(self, url: str, base_domain: str) -> bool:
        """Check if URL should be crawled"""
        try:
            parsed = urlparse(url)
            
            # Must be same domain
            if parsed.netloc != base_domain:
                self.debug_info.append(f"Skipping different domain: {url} (expected {base_domain})")
                return False
            
            # Skip common non-content URLs
            skip_extensions = {'.pdf', '.jpg', '.jpeg', '.png', '.gif', '.css', '.js', '.ico', '.svg', '.xml', '.mp4', '.zip', '.gz'}
            if any(url.lower().endswith(ext) for ext in skip_extensions):
                self.debug_info.append(f"Skipping file extension: {url}")
                return False
            
            # Skip common non-content paths
            skip_paths = {'wp-admin', 'admin', 'login', 'register', 'cart', 'checkout', 'wp-json', 'feed', 'xmlrpc.php'}
            if any(path in url.lower() for path in skip_paths):
                self.debug_info.append(f"Skipping excluded path: {url}")
                return False
            
            # Skip URLs with query parameters (optional, can be commented out if needed)
            # if '?' in url:
            #     self.debug_info.append(f"Skipping URL with query params: {url}")
            #     return False
            
            # Skip URLs with fragments (optional, can be commented out if needed)
            # if '#' in url:
            #     self.debug_info.append(f"Skipping URL with fragment: {url}")
            #     return False
            
            return True
        except Exception as e:
            self.debug_info.append(f"Error validating URL {url}: {str(e)}")
            return False
    
    def extract_links(self, html: str, base_url: str) -> Set[str]:
        """Extract all links from HTML"""
        links = set()
        soup = BeautifulSoup(html, 'html.parser')
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            # Skip empty or javascript links
            if not href or href.startswith('javascript:') or href == '#':
                continue
                
            # Convert relative URLs to absolute
            full_url = urljoin(base_url, href)
            
            # Remove fragments
            if '#' in full_url:
                full_url = full_url.split('#')[0]
                
            # Remove trailing slashes for consistency
            if full_url.endswith('/'):
                full_url = full_url[:-1]
                
            links.add(full_url)
        
        self.debug_info.append(f"Extracted {len(links)} links from {base_url}")
        return links
    
    def crawl_page(self, url: str) -> Dict:
        """Crawl a single page and return its data"""
        try:
            self.debug_info.append(f"Crawling page: {url}")
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title = soup.find('title')
            title = title.get_text().strip() if title else "Untitled"
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract text content
            content = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in content.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            content = ' '.join(chunk for chunk in chunks if chunk)
            
            # Limit content length
            if len(content) > config.MAX_CONTENT_LENGTH:
                content = content[:config.MAX_CONTENT_LENGTH] + "..."
            
            # Extract links for further crawling
            links = self.extract_links(response.text, url)
            
            self.debug_info.append(f"Successfully processed {url}, found {len(links)} links")
            
            return {
                'url': url,
                'title': title,
                'content': content,
                'word_count': len(content.split()),
                'status': 'success',
                'links': links
            }
            
        except Exception as e:
            self.debug_info.append(f"Error crawling {url}: {str(e)}")
            return {
                'url': url,
                'title': f"Error: {str(e)[:50]}...",
                'content': '',
                'word_count': 0,
                'status': f'error: {str(e)}',
                'links': set()
            }
    
    def crawl_site(self, start_url: str, progress_callback=None) -> List[Dict]:
        """Crawl an entire site starting from the given URL"""
        # Reset for a fresh crawl
        self.visited_urls = set()
        self.found_urls = set()
        self.debug_info = []
        crawled_pages = []
        
        # Ensure the start URL has a scheme
        if not start_url.startswith(('http://', 'https://')):
            start_url = 'https://' + start_url
            self.debug_info.append(f"Added https:// to URL: {start_url}")
        
        # Normalize the start URL
        if start_url.endswith('/'):
            start_url = start_url[:-1]
            
        # Parse the base domain
        try:
            parsed_url = urlparse(start_url)
            base_domain = parsed_url.netloc
            
            if not base_domain:
                self.debug_info.append(f"Invalid URL format: {start_url}")
                return []
                
            self.debug_info.append(f"Starting crawl of {start_url}, base domain: {base_domain}")
        except Exception as e:
            self.debug_info.append(f"Error parsing URL {start_url}: {str(e)}")
            return []
        
        # Add the start URL to the queue
        self.found_urls.add(start_url)
        
        # Create a progress bar if we have a callback
        if progress_callback:
            progress_callback(0, f"Starting crawl of {start_url}")
        
        # Main crawling loop
        while self.found_urls and len(crawled_pages) < self.max_pages:
            # Update progress
            if progress_callback:
                progress = len(crawled_pages) / self.max_pages
                progress_callback(progress, f"Crawled {len(crawled_pages)}/{self.max_pages} pages")
            
            # Get next URL to crawl
            current_url = self.found_urls.pop()
            self.debug_info.append(f"Processing: {current_url} (Queue size: {len(self.found_urls)})")
            
            # Skip if already visited
            if current_url in self.visited_urls:
                self.debug_info.append(f"Skipping already visited: {current_url}")
                continue
            
            # Check if URL is valid for crawling
            if not self.is_valid_url(current_url, base_domain):
                self.debug_info.append(f"Invalid URL: {current_url}")
                continue
            
            # Mark as visited before crawling to prevent duplicates in case of errors
            self.visited_urls.add(current_url)
            
            # Crawl the page
            page_data = self.crawl_page(current_url)
            crawled_pages.append(page_data)
            
            # Add new links to crawl queue
            if page_data['status'] == 'success':
                new_links_count = 0
                for link in page_data['links']:
                    if link not in self.visited_urls and link not in self.found_urls:
                        self.found_urls.add(link)
                        new_links_count += 1
                
                self.debug_info.append(f"Added {new_links_count} new URLs to queue")
                self.debug_info.append(f"Queue now contains {len(self.found_urls)} URLs")
                self.debug_info.append(f"Visited URLs: {len(self.visited_urls)}")
            
            # Be respectful - delay between requests
            time.sleep(self.delay)
        
        # Final progress update
        if progress_callback:
            progress_callback(1.0, f"Completed crawl - {len(crawled_pages)} pages processed")
        
        self.debug_info.append(f"Crawl complete. Processed {len(crawled_pages)} pages out of {self.max_pages} maximum.")
        
        return crawled_pages
    
    def get_debug_info(self) -> List[str]:
        """Get debug information about the crawl"""
        return self.debug_info