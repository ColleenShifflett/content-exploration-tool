import requests
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import time
from typing import Set, List, Dict
import config

class SiteCrawler:
    def __init__(self, max_pages: int = 5, delay: float = 1.0):  # Changed default from 50 to 5
        self.max_pages = max_pages
        self.delay = delay  # Delay between requests to be respectful
        self.visited_urls: Set[str] = set()
        self.found_urls: Set[str] = set()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
    
    def is_valid_url(self, url: str, base_domain: str) -> bool:
        """Check if URL should be crawled"""
        try:
            parsed = urlparse(url)
            
            # Must be same domain
            if parsed.netloc != base_domain:
                return False
            
            # Skip common non-content URLs
            skip_extensions = {'.pdf', '.jpg', '.jpeg', '.png', '.gif', '.css', '.js', '.ico', '.svg'}
            if any(url.lower().endswith(ext) for ext in skip_extensions):
                return False
            
            # Skip common non-content paths
            skip_paths = {'wp-admin', 'admin', 'login', 'register', 'cart', 'checkout'}
            if any(path in url.lower() for path in skip_paths):
                return False
            
            return True
        except:
            return False
    
    def extract_links(self, html: str, base_url: str) -> Set[str]:
        """Extract all links from HTML"""
        links = set()
        soup = BeautifulSoup(html, 'html.parser')
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            # Convert relative URLs to absolute
            full_url = urljoin(base_url, href)
            # Remove fragments
            full_url = full_url.split('#')[0]
            links.add(full_url)
        
        return links
    
    def crawl_page(self, url: str) -> Dict:
        """Crawl a single page and return its data"""
        try:
            print(f"Crawling: {url}")
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
            
            return {
                'url': url,
                'title': title,
                'content': content,
                'word_count': len(content.split()),
                'status': 'success',
                'links': links
            }
            
        except Exception as e:
            return {
                'url': url,
                'title': 'Error',
                'content': '',
                'word_count': 0,
                'status': f'error: {str(e)}',
                'links': set()
            }
    
    def crawl_site(self, start_url: str) -> List[Dict]:
        """Crawl an entire site starting from the given URL"""
        base_domain = urlparse(start_url).netloc
        self.found_urls.add(start_url)
        crawled_pages = []
        
        while self.found_urls and len(crawled_pages) < self.max_pages:
            # Get next URL to crawl
            current_url = self.found_urls.pop()
            
            if current_url in self.visited_urls:
                continue
            
            if not self.is_valid_url(current_url, base_domain):
                continue
            
            # Crawl the page
            page_data = self.crawl_page(current_url)
            crawled_pages.append(page_data)
            self.visited_urls.add(current_url)
            
            # Add new links to crawl queue
            if page_data['status'] == 'success':
                for link in page_data['links']:
                    if link not in self.visited_urls:
                        self.found_urls.add(link)
            
            # Be respectful - delay between requests
            time.sleep(self.delay)
        
        return crawled_pages
