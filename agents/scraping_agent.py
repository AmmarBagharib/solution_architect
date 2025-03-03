"""
Scraping Agent module that collects pricing information from cloud providers.
"""
import asyncio
import json
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

# This is a placeholder for the actual crawl4ai import
# We'll need to adjust based on the actual API
try:
    from crawl4ai import Crawler, WebPage
except ImportError:
    # Mock implementation for development purposes
    class WebPage:
        def __init__(self, url: str, html: str = "", markdown: str = ""):
            self.url = url
            self.html = html
            self.markdown = markdown

    class Crawler:
        async def crawl(self, urls: List[str]) -> List[WebPage]:
            return [WebPage(url=url) for url in urls]

import pandas as pd
from bs4 import BeautifulSoup
import requests

from .base_agent import BaseAgent


class ScrapingAgent(BaseAgent):
    """
    Agent responsible for scraping cloud provider websites for pricing information.
    
    This agent uses crawl4ai to collect pricing data and implements a caching
    mechanism to avoid repeated scraping of the same data.
    """
    
    def __init__(self, cache_dir: str = "data/cache", cache_ttl_hours: int = 24):
        """
        Initialize the scraping agent with caching parameters.
        
        Args:
            cache_dir: Directory where cached data will be stored.
            cache_ttl_hours: Time-to-live for cached data in hours.
        """
        super().__init__(name="ScrapingAgent")
        self.cache_dir = cache_dir
        self.cache_ttl_hours = cache_ttl_hours
        self.crawler = Crawler()
        
        # Ensure cache directory exists
        os.makedirs(cache_dir, exist_ok=True)
        
        # Default cloud provider URLs
        self.provider_urls = {
            "aws": {
                "rds": "https://aws.amazon.com/rds/pricing/",
                "dynamodb": "https://aws.amazon.com/dynamodb/pricing/",
                "s3": "https://aws.amazon.com/s3/pricing/",
            },
            "gcp": {
                "cloud_sql": "https://cloud.google.com/sql/pricing",
                "firestore": "https://cloud.google.com/firestore/pricing",
                "cloud_storage": "https://cloud.google.com/storage/pricing",
            },
            "azure": {
                "sql_database": "https://azure.microsoft.com/en-us/pricing/details/sql-database/single/",
                "cosmos_db": "https://azure.microsoft.com/en-us/pricing/details/cosmos-db/",
                "blob_storage": "https://azure.microsoft.com/en-us/pricing/details/storage/blobs/",
            }
        }
    
    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the scraping operation based on user requirements.
        
        Args:
            input_data: Contains the user's requirements including preferred providers.
            
        Returns:
            Dictionary containing the scraped pricing data.
        """
        self.log_info("Starting scraping operation")
        
        # Extract relevant information from input
        preferred_providers = input_data.get("preferred_providers", list(self.provider_urls.keys()))
        file_types = input_data.get("file_types", [])
        
        # Determine which URLs to scrape
        urls_to_scrape = self._determine_urls_to_scrape(preferred_providers, file_types)
        
        # Check cache first
        cached_data = self._get_cached_data(urls_to_scrape)
        
        # Filter out URLs that we already have cached data for
        urls_to_scrape = [url for url in urls_to_scrape if url not in cached_data]
        
        # Scrape remaining URLs
        if urls_to_scrape:
            self.log_info(f"Scraping {len(urls_to_scrape)} URLs")
            scraped_data = await self._scrape_urls(urls_to_scrape)
            
            # Cache new data
            self._cache_data(scraped_data)
            
            # Merge with cached data
            all_data = {**cached_data, **scraped_data}
        else:
            self.log_info("All data available from cache")
            all_data = cached_data
        
        # Process and normalize the data
        processed_data = self._process_data(all_data)
        
        return {
            "pricing_data": processed_data,
            "scraped_at": datetime.now().isoformat()
        }
    
    def _determine_urls_to_scrape(self, 
                               preferred_providers: List[str], 
                               file_types: List[str]) -> List[str]:
        """
        Determine which URLs to scrape based on user preferences.
        
        Args:
            preferred_providers: List of preferred cloud providers.
            file_types: List of file types the user intends to store.
            
        Returns:
            List of URLs to scrape.
        """
        urls = []
        
        # Map file types to database services
        service_mapping = {
            "pdf": ["s3", "cloud_storage", "blob_storage"],
            "cache": ["dynamodb", "firestore", "cosmos_db"],
            "structured_data": ["rds", "cloud_sql", "sql_database"],
            # Add other mappings as needed
        }
        
        # Default to scrape all if no file types specified
        if not file_types:
            for provider in preferred_providers:
                if provider in self.provider_urls:
                    for service_url in self.provider_urls[provider].values():
                        urls.append(service_url)
        else:
            # Scrape specific services based on file types
            for file_type in file_types:
                relevant_services = service_mapping.get(file_type, [])
                for provider in preferred_providers:
                    if provider in self.provider_urls:
                        for service, url in self.provider_urls[provider].items():
                            # Check if service is relevant for this file type
                            if any(relevant_service in service for relevant_service in relevant_services):
                                urls.append(url)
        
        return list(set(urls))  # Remove duplicates
    
    async def _scrape_urls(self, urls: List[str]) -> Dict[str, str]:
        """
        Scrape a list of URLs using crawl4ai.
        
        Args:
            urls: List of URLs to scrape.
            
        Returns:
            Dictionary mapping URLs to their markdown content.
        """
        try:
            # Use crawl4ai to scrape the URLs
            web_pages = await self.crawler.crawl(urls)
            
            # Create a dictionary of URL to markdown content
            return {page.url: page.markdown for page in web_pages}
            
        except Exception as e:
            self.log_error(f"Error during scraping: {str(e)}")
            
            # Fallback to a simpler method if crawl4ai fails
            results = {}
            for url in urls:
                try:
                    response = requests.get(url)
                    soup = BeautifulSoup(response.text, 'html.parser')
                    # Extract just the text content as a simple fallback
                    text = soup.get_text(separator=" ", strip=True)
                    results[url] = text
                except Exception as inner_e:
                    self.log_error(f"Failed to scrape {url}: {str(inner_e)}")
                    results[url] = ""
            
            return results
    
    def _get_cached_data(self, urls: List[str]) -> Dict[str, str]:
        """
        Retrieve cached data for the given URLs if available and not expired.
        
        Args:
            urls: List of URLs to check in the cache.
            
        Returns:
            Dictionary of cached data for the URLs.
        """
        cached_data = {}
        cache_ttl = timedelta(hours=self.cache_ttl_hours)
        
        for url in urls:
            cache_file = os.path.join(self.cache_dir, self._url_to_filename(url))
            if os.path.exists(cache_file):
                # Check if the cache is still valid
                modified_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
                if datetime.now() - modified_time < cache_ttl:
                    try:
                        with open(cache_file, 'r', encoding='utf-8') as f:
                            cache_entry = json.load(f)
                            cached_data[url] = cache_entry['content']
                            self.log_info(f"Retrieved {url} from cache")
                    except Exception as e:
                        self.log_error(f"Error reading cache for {url}: {str(e)}")
        
        return cached_data
    
    def _cache_data(self, data: Dict[str, str]) -> None:
        """
        Cache the scraped data with timestamps.
        
        Args:
            data: Dictionary mapping URLs to their content.
        """
        for url, content in data.items():
            cache_file = os.path.join(self.cache_dir, self._url_to_filename(url))
            try:
                cache_entry = {
                    'url': url,
                    'content': content,
                    'timestamp': datetime.now().isoformat()
                }
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(cache_entry, f, ensure_ascii=False, indent=2)
                self.log_info(f"Cached data for {url}")
            except Exception as e:
                self.log_error(f"Error caching data for {url}: {str(e)}")
    
    def _url_to_filename(self, url: str) -> str:
        """
        Convert a URL to a valid filename for caching.
        
        Args:
            url: The URL to convert.
            
        Returns:
            A safe filename based on the URL.
        """
        # Replace special characters with underscores
        safe_name = "".join([c if c.isalnum() else "_" for c in url])
        # Add md5 hash to prevent filename collisions and length issues
        import hashlib
        url_hash = hashlib.md5(url.encode()).hexdigest()
        return f"{url_hash}_{safe_name[-50:]}.json"
    
    def _process_data(self, data: Dict[str, str]) -> Dict[str, Any]:
        """
        Process and normalize the scraped data for easier comparison.
        
        Args:
            data: Dictionary mapping URLs to their content.
            
        Returns:
            Processed and structured pricing data.
        """
        processed_data = {
            "storage_pricing": {},
            "database_pricing": {},
            "compute_pricing": {},
            "raw_data": data
        }
        
        # For each provider and service, extract pricing information
        for url, content in data.items():
            # Determine provider and service type from URL
            provider, service_type = self._identify_provider_and_service(url)
            
            if provider and service_type:
                # Extract pricing information based on provider and service type
                pricing_info = self._extract_pricing_info(provider, service_type, content)
                
                # Add to appropriate category
                if service_type in ["s3", "cloud_storage", "blob_storage"]:
                    processed_data["storage_pricing"][f"{provider}_{service_type}"] = pricing_info
                elif service_type in ["rds", "cloud_sql", "sql_database", "dynamodb", "firestore", "cosmos_db"]:
                    processed_data["database_pricing"][f"{provider}_{service_type}"] = pricing_info
                else:
                    processed_data["compute_pricing"][f"{provider}_{service_type}"] = pricing_info
        
        return processed_data
    
    def _identify_provider_and_service(self, url: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Identify the cloud provider and service from a URL.
        
        Args:
            url: The URL to analyze.
            
        Returns:
            Tuple of (provider, service) or (None, None) if not identified.
        """
        for provider, services in self.provider_urls.items():
            for service, service_url in services.items():
                if service_url in url:
                    return provider, service
        
        # Try harder to identify provider if exact match not found
        if "aws.amazon.com" in url:
            provider = "aws"
        elif "cloud.google.com" in url:
            provider = "gcp"
        elif "azure.microsoft.com" in url:
            provider = "azure"
        else:
            return None, None
        
        # Try to identify service type
        if "database" in url or "db" in url:
            return provider, "database"
        elif "storage" in url:
            return provider, "storage"
        elif "compute" in url:
            return provider, "compute"
        
        return provider, None
    
    def _extract_pricing_info(self, 
                           provider: str, 
                           service_type: str, 
                           content: str) -> Dict[str, Any]:
        """
        Extract structured pricing information from the scraped content.
        
        This is a placeholder implementation and would need to be customized
        for each provider and service type based on their pricing page structure.
        
        Args:
            provider: The cloud provider.
            service_type: The type of service.
            content: The scraped content.
            
        Returns:
            Structured pricing information.
        """
        # This is a simplified mock implementation
        # In a real implementation, we would use more sophisticated parsing
        pricing_info = {
            "provider": provider,
            "service": service_type,
            "pricing_tiers": [],
            "free_tier": None,
            "egress_costs": {},
            "storage_costs": {},
            "operation_costs": {},
        }
        
        # For demonstration purposes, we'll add some mock data
        # In a real implementation, we would parse the content
        if "storage" in service_type:
            pricing_info["storage_costs"] = {
                "standard": {
                    "first_50_tb": 0.023,  # USD per GB per month
                    "next_450_tb": 0.022,
                    "over_500_tb": 0.021
                }
            }
            pricing_info["egress_costs"] = {
                "same_region": 0.00,
                "different_region": 0.01,
                "internet": 0.12
            }
        elif "database" in service_type:
            pricing_info["pricing_tiers"] = [
                {"name": "Basic", "price": 5.00, "storage": "5GB", "connections": "60"},
                {"name": "Standard", "price": 15.00, "storage": "20GB", "connections": "125"},
                {"name": "Premium", "price": 40.00, "storage": "50GB", "connections": "250"}
            ]
        
        # Check for free tier mentions
        if "free tier" in content.lower() or "free usage" in content.lower():
            pricing_info["free_tier"] = True
        
        return pricing_info