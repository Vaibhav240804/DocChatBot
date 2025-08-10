import requests
from bs4 import BeautifulSoup
import trafilatura
import re
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse
import json
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from config import Config

class MetadataExtractor:
    def __init__(self):
        self.llm = ChatGroq(
            groq_api_key=Config.GROQ_API_KEY,
            model_name=Config.GROQ_MODEL,
            temperature=0.1
        )
        
        self.metadata_prompt = ChatPromptTemplate.from_template("""
        You are an expert at extracting structured metadata from technical documentation.
        
        Given the following HTML content and extracted text from a technical document, 
        extract the following metadata in JSON format:
        
        HTML Content: {html_content}
        Text Content: {text_content}
        URL: {url}
        
        Extract:
        1. title: The main title of the document
        2. author: Author name(s) if mentioned
        3. department: Department or organization if mentioned  
        4. topic: Main topic/subject area
        5. prerequisites: Any prerequisites mentioned (as a list)
        6. redirects: Any links or redirects to other sections/pages (as a list)
        7. section: Current section name if this is part of a larger document
        8. subsection: Subsection name if applicable
        
        For prerequisites, look for phrases like:
        - "Before you begin"
        - "Prerequisites" 
        - "You should know"
        - "Required knowledge"
        - "Dependencies"
        
        For redirects, extract any internal links or references to other documentation pages.
        
        Return ONLY valid JSON with these exact keys. If information is not found, use null or empty array.
        """)

    def scrape_url(self, url: str) -> Tuple[str, str, Dict]:
        """Scrape URL and return HTML content, text content, and basic metadata"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=Config.REQUEST_TIMEOUT)
            response.raise_for_status()
            
            # Get clean text using trafilatura
            text_content = trafilatura.extract(response.text) or ""
            
            # Parse HTML with BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
                
            # Get HTML content (limited for LLM processing)
            html_content = str(soup)[:3000]  # Limit HTML size for LLM
            
            # Extract basic metadata
            basic_metadata = self._extract_basic_metadata(soup, url)
            
            return html_content, text_content, basic_metadata
            
        except Exception as e:
            print(f"Error scraping {url}: {str(e)}")
            return "", "", {}

    def _extract_basic_metadata(self, soup: BeautifulSoup, url: str) -> Dict:
        """Extract basic metadata from HTML using traditional methods"""
        metadata = {}
        
        # Title
        title_tag = soup.find('title')
        if title_tag:
            metadata['title'] = title_tag.get_text().strip()
        else:
            h1_tag = soup.find('h1')
            metadata['title'] = h1_tag.get_text().strip() if h1_tag else ""
        
        # Meta tags
        meta_tags = soup.find_all('meta')
        for tag in meta_tags:
            if tag.get('name') == 'author':
                metadata['author'] = tag.get('content', '')
            elif tag.get('name') == 'description':
                metadata['description'] = tag.get('content', '')
            elif tag.get('property') == 'article:author':
                metadata['author'] = tag.get('content', '')
        
        # URL
        metadata['url'] = url
        
        # Extract internal links
        links = []
        for link in soup.find_all('a', href=True):
            href = link.get('href')
            if href and not href.startswith(('http://', 'https://', 'mailto:', '#')):
                full_url = urljoin(url, href)
                links.append(full_url)
        metadata['internal_links'] = links[:10]  # Limit to first 10
        
        return metadata

    def extract_metadata_with_llm(self, html_content: str, text_content: str, url: str) -> Dict:
        """Use LLM to extract sophisticated metadata"""
        try:
            # Limit content size for LLM
            limited_html = html_content[:2000]
            limited_text = text_content[:3000]
            
            response = self.llm.invoke(
                self.metadata_prompt.format(
                    html_content=limited_html,
                    text_content=limited_text,
                    url=url
                )
            )
            
            # Clean the response content - sometimes there's extra text
            content = response.content.strip()
            
            # Find JSON in the response
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = content[json_start:json_end]
                metadata = json.loads(json_str)
            else:
                # Fallback if no JSON found
                raise json.JSONDecodeError("No valid JSON found", content, 0)
            
            # Clean and validate metadata
            cleaned_metadata = {}
            for field in Config.METADATA_FIELDS:
                value = metadata.get(field)
                
                if field in ['prerequisites', 'redirects']:
                    # Ensure these are lists and convert to strings for vector store
                    if isinstance(value, list):
                        cleaned_metadata[field] = [str(item) for item in value if item] if value else []
                    elif value:
                        cleaned_metadata[field] = [str(value)]
                    else:
                        cleaned_metadata[field] = []
                else:
                    # Convert other fields to strings or None
                    cleaned_metadata[field] = str(value) if value is not None else None
            
            return cleaned_metadata
            
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing LLM metadata response: {str(e)}")
            # Return fallback metadata
            return self._create_fallback_metadata()
        except Exception as e:
            print(f"Error extracting metadata with LLM: {str(e)}")
            return self._create_fallback_metadata()

    def _create_fallback_metadata(self) -> Dict:
        """Create fallback metadata when LLM extraction fails"""
        return {
            'title': None,
            'author': None,
            'department': None,
            'topic': None,
            'prerequisites': [],
            'redirects': [],
            'section': None,
            'subsection': None
        }

    def extract_full_metadata(self, url: str) -> Dict:
        """Extract complete metadata combining traditional and LLM methods"""
        # Scrape the URL
        html_content, text_content, basic_metadata = self.scrape_url(url)
        
        if not text_content:
            print(f"No content extracted from {url}")
            return basic_metadata
        
        # Extract metadata with LLM
        llm_metadata = self.extract_metadata_with_llm(html_content, text_content, url)
        
        # Merge metadata (LLM takes precedence, but fill gaps with basic)
        final_metadata = basic_metadata.copy()
        final_metadata.update({k: v for k, v in llm_metadata.items() if v})
        
        # Ensure URL is always present
        final_metadata['url'] = url
        final_metadata['text_content'] = text_content
        
        return final_metadata

    def extract_sections(self, text_content: str, metadata: Dict) -> List[Dict]:
        """Split content into sections with metadata"""
        sections = []
        
        # Simple section splitting based on headers
        lines = text_content.split('\n')
        current_section = []
        current_title = metadata.get('title', 'Unknown')
        section_count = 0
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if line is a header (simple heuristic)
            if (len(line) < 100 and 
                (line.isupper() or 
                 re.match(r'^#+\s+', line) or 
                 re.match(r'^\d+\.', line) or
                 line.endswith(':'))):
                
                # Save previous section
                if current_section and len('\n'.join(current_section).strip()) > 100:
                    section_metadata = metadata.copy()
                    section_metadata['section'] = current_title
                    section_metadata['text_content'] = '\n'.join(current_section)
                    section_metadata['section_id'] = f"{metadata.get('url', '')}#{section_count}"
                    sections.append(section_metadata)
                    section_count += 1
                
                # Start new section
                current_section = [line]
                current_title = line.strip('#').strip()
            else:
                current_section.append(line)
        
        # Add final section
        if current_section and len('\n'.join(current_section).strip()) > 100:
            section_metadata = metadata.copy()
            section_metadata['section'] = current_title
            section_metadata['text_content'] = '\n'.join(current_section)
            section_metadata['section_id'] = f"{metadata.get('url', '')}#{section_count}"
            sections.append(section_metadata)
        
        # If no sections found, return the whole document
        if not sections:
            metadata['section'] = metadata.get('title', 'Main Content')
            metadata['section_id'] = f"{metadata.get('url', '')}#0"
            sections.append(metadata)
        
        return sections