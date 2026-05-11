"""
LinkedIn Financial News Scraper
================================

This module scrapes financial news and posts from LinkedIn using the LinkedIn API.
Focuses on posts from financial influencers, companies, and industry leaders.

Key Features:
- OAuth2 authentication
- Rate limiting and request throttling
- Company page and influencer monitoring
- Content extraction and ticker identification
- Engagement metrics tracking

Note: LinkedIn API has strict rate limits (100 requests per user per day for Community Management API)
For production, consider using official partnerships or Data Alliance access.

Author: TradeBrains Team
"""

import logging
import time
import hashlib
import re
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
import json

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import redis
from kafka import KafkaProducer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class LinkedInPost:
    """Structured LinkedIn post data model"""
    post_id: str
    author_id: str
    author_name: str
    author_headline: str
    text: str
    created_at: str
    tickers: List[str]
    likes: int
    comments: int
    shares: int
    impressions: int
    author_followers: int
    is_company_post: bool
    company_name: Optional[str]
    content_hash: str
    source: str = "linkedin"


class LinkedInScraper:
    """
    LinkedIn API client for financial news scraping.

    Architecture Decision:
    - Uses Official LinkedIn API v2 (requires approval)
    - OAuth2 for authentication
    - Rate limiting with token bucket algorithm
    - Focus on high-quality, professional content

    Why LinkedIn?
    - Professional network with verified users
    - High-quality financial analysis and news
    - Company announcements and earnings updates
    - Less noise compared to Twitter

    Limitations:
    - Strict API rate limits (100 req/day for basic access)
    - Requires company approval for higher limits
    - Limited public post access without user tokens
    """

    BASE_URL = "https://api.linkedin.com/v2"
    AUTH_URL = "https://www.linkedin.com/oauth/v2/authorization"
    TOKEN_URL = "https://www.linkedin.com/oauth/v2/accessToken"

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        access_token: str,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        kafka_bootstrap_servers: str = "localhost:9092"
    ):
        """
        Initialize LinkedIn scraper.

        Args:
            client_id: LinkedIn app client ID
            client_secret: LinkedIn app client secret
            access_token: OAuth2 access token
            redis_host: Redis server for deduplication and rate limiting
            redis_port: Redis port
            kafka_bootstrap_servers: Kafka broker addresses
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.access_token = access_token

        # Create session with retry logic
        self.session = self._create_session()

        # Redis for deduplication and rate limiting
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=0,
            decode_responses=True
        )

        # Kafka producer
        self.kafka_producer = KafkaProducer(
            bootstrap_servers=kafka_bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            compression_type='gzip',
            acks='all',
            retries=3
        )

        # Rate limiting (100 requests per day = ~0.0012 req/sec)
        # Using token bucket algorithm
        self.rate_limit_key = "linkedin:rate_limit"
        self.max_requests_per_day = 100
        self.rate_limit_window = 86400  # 24 hours in seconds

        # Ticker extraction
        self.ticker_pattern = re.compile(r'\$[A-Z]{1,5}\b')

        logger.info("LinkedInScraper initialized")

    def _create_session(self) -> requests.Session:
        """
        Create requests session with retry logic.

        Returns:
            Configured requests.Session
        """
        session = requests.Session()

        # Retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=2,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"]
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.mount("http://", adapter)

        # Default headers
        session.headers.update({
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
            "X-Restli-Protocol-Version": "2.0.0"
        })

        return session

    def _check_rate_limit(self) -> bool:
        """
        Check if we're within rate limits using token bucket.

        Returns:
            True if request allowed, False if rate limited
        """
        current_count = self.redis_client.get(self.rate_limit_key)

        if current_count is None:
            # First request in window
            self.redis_client.set(
                self.rate_limit_key,
                1,
                ex=self.rate_limit_window
            )
            return True

        current_count = int(current_count)

        if current_count >= self.max_requests_per_day:
            logger.warning("LinkedIn rate limit reached")
            return False

        # Increment counter
        self.redis_client.incr(self.rate_limit_key)
        return True

    def _extract_tickers(self, text: str) -> List[str]:
        """Extract stock tickers from text."""
        tickers = self.ticker_pattern.findall(text.upper())
        return list(set([ticker[1:] for ticker in tickers]))

    def _compute_content_hash(self, text: str) -> str:
        """Generate content hash for deduplication."""
        normalized = re.sub(r'http\S+|\s+', ' ', text.lower()).strip()
        return hashlib.sha256(normalized.encode()).hexdigest()

    def _is_duplicate(self, content_hash: str) -> bool:
        """Check if post is duplicate."""
        key = f"linkedin:hash:{content_hash}"
        is_new = self.redis_client.set(key, "1", nx=True, ex=30*24*60*60)
        return not is_new

    def get_organization_posts(self, organization_id: str, count: int = 10) -> List[Dict]:
        """
        Fetch posts from a company/organization page.

        Args:
            organization_id: LinkedIn organization URN
            count: Number of posts to fetch (max 50)

        Returns:
            List of post dictionaries
        """
        if not self._check_rate_limit():
            logger.warning("Rate limit exceeded, skipping request")
            return []

        try:
            url = f"{self.BASE_URL}/shares"
            params = {
                "q": "owners",
                "owners": f"urn:li:organization:{organization_id}",
                "count": min(count, 50),
                "sortBy": "LAST_MODIFIED"
            }

            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            posts = data.get("elements", [])

            logger.info(f"Fetched {len(posts)} posts from organization {organization_id}")
            return posts

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch organization posts: {e}")
            return []

    def get_user_posts(self, person_urn: str, count: int = 10) -> List[Dict]:
        """
        Fetch posts from a user profile.

        Note: Requires user's permission or connection

        Args:
            person_urn: LinkedIn person URN
            count: Number of posts to fetch

        Returns:
            List of post dictionaries
        """
        if not self._check_rate_limit():
            logger.warning("Rate limit exceeded, skipping request")
            return []

        try:
            url = f"{self.BASE_URL}/shares"
            params = {
                "q": "owners",
                "owners": person_urn,
                "count": min(count, 50),
                "sortBy": "LAST_MODIFIED"
            }

            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            posts = data.get("elements", [])

            logger.info(f"Fetched {len(posts)} posts from user {person_urn}")
            return posts

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch user posts: {e}")
            return []

    def process_post(self, post_data: Dict) -> Optional[LinkedInPost]:
        """
        Process and validate LinkedIn post.

        Args:
            post_data: Raw post data from API

        Returns:
            Structured LinkedInPost or None
        """
        try:
            # Extract text content
            text_content = post_data.get("text", {}).get("text", "")
            if not text_content:
                return None

            # Extract tickers
            tickers = self._extract_tickers(text_content)

            # Filter: Only posts with tickers or financial keywords
            financial_keywords = ['earnings', 'revenue', 'market', 'stock', 'shares', 'dividend']
            has_financial_content = any(kw in text_content.lower() for kw in financial_keywords)

            if not tickers and not has_financial_content:
                return None

            # Compute hash
            content_hash = self._compute_content_hash(text_content)

            # Check duplicates
            if self._is_duplicate(content_hash):
                return None

            # Extract author information
            owner = post_data.get("owner", "")
            is_company = "organization" in owner

            # Extract engagement metrics
            engagement = post_data.get("content", {}).get("shareStatistics", {})

            # Structure post
            linkedin_post = LinkedInPost(
                post_id=post_data.get("id", ""),
                author_id=owner,
                author_name=post_data.get("author", {}).get("name", "Unknown"),
                author_headline=post_data.get("author", {}).get("headline", ""),
                text=text_content,
                created_at=datetime.fromtimestamp(
                    post_data.get("created", {}).get("time", 0) / 1000
                ).isoformat(),
                tickers=tickers,
                likes=engagement.get("likeCount", 0),
                comments=engagement.get("commentCount", 0),
                shares=engagement.get("shareCount", 0),
                impressions=engagement.get("impressionCount", 0),
                author_followers=post_data.get("author", {}).get("followerCount", 0),
                is_company_post=is_company,
                company_name=post_data.get("author", {}).get("name") if is_company else None,
                content_hash=content_hash
            )

            logger.info(f"Processed LinkedIn post {linkedin_post.post_id}")
            return linkedin_post

        except Exception as e:
            logger.error(f"Error processing post: {e}", exc_info=True)
            return None

    def publish_to_kafka(self, post: LinkedInPost):
        """Publish post to Kafka."""
        try:
            self.kafka_producer.send(
                topic='raw_linkedin',
                value=asdict(post),
                key=post.post_id.encode('utf-8')
            )
            logger.debug(f"Published post {post.post_id} to Kafka")

        except Exception as e:
            logger.error(f"Failed to publish to Kafka: {e}")

    def monitor_organizations(self, org_ids: List[str], interval: int = 3600):
        """
        Monitor multiple organizations continuously.

        Args:
            org_ids: List of LinkedIn organization IDs
            interval: Polling interval in seconds (default 1 hour)
        """
        logger.info(f"Monitoring {len(org_ids)} organizations")

        while True:
            for org_id in org_ids:
                try:
                    posts = self.get_organization_posts(org_id)

                    for post_data in posts:
                        processed = self.process_post(post_data)
                        if processed:
                            self.publish_to_kafka(processed)

                    # Avoid rate limiting between requests
                    time.sleep(2)

                except Exception as e:
                    logger.error(f"Error monitoring org {org_id}: {e}")

            logger.info(f"Sleeping for {interval} seconds...")
            time.sleep(interval)


# Example usage
if __name__ == "__main__":
    """
    Production deployment example.

    Environment Variables:
    - LINKEDIN_CLIENT_ID
    - LINKEDIN_CLIENT_SECRET
    - LINKEDIN_ACCESS_TOKEN
    - REDIS_HOST
    - KAFKA_BOOTSTRAP_SERVERS
    """
    import os

    CLIENT_ID = os.getenv("LINKEDIN_CLIENT_ID")
    CLIENT_SECRET = os.getenv("LINKEDIN_CLIENT_SECRET")
    ACCESS_TOKEN = os.getenv("LINKEDIN_ACCESS_TOKEN")
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    KAFKA_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")

    if not all([CLIENT_ID, CLIENT_SECRET, ACCESS_TOKEN]):
        raise ValueError("LinkedIn credentials required")

    scraper = LinkedInScraper(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        access_token=ACCESS_TOKEN,
        redis_host=REDIS_HOST,
        kafka_bootstrap_servers=KAFKA_SERVERS
    )

    # Monitor financial companies and influencers
    # Replace with actual LinkedIn organization IDs
    ORGANIZATIONS = [
        "1234567",  # Example: Bloomberg
        "2345678",  # Example: CNBC
        "3456789",  # Example: Goldman Sachs
    ]

    scraper.monitor_organizations(ORGANIZATIONS, interval=3600)


"""
INTERVIEW PREPARATION NOTES
===========================

Q: Why LinkedIn over other news sources?
A: 1) Professional network with verified users
   2) High-quality analysis from industry experts
   3) Direct company announcements
   4) Less noise/spam than Twitter

Q: How do you handle strict rate limits (100 req/day)?
A: 1) Token bucket algorithm with Redis
   2) Prioritize high-value organizations
   3) Cache responses
   4) Consider LinkedIn Data Alliance for higher limits

Q: What if access token expires?
A: 1) Implement OAuth2 refresh token flow
   2) Store refresh token securely
   3) Auto-refresh before expiration
   4) Alert on auth failures

Q: How do you identify financial influencers?
A: 1) Manual curation of known analysts
   2) Track engagement metrics (likes, shares)
   3) Keyword analysis of profile headlines
   4) Connection to financial organizations

Q: Can you scrape without API?
A: Technically yes (selenium/playwright), but:
   1) Violates LinkedIn ToS
   2) High risk of account ban
   3) Not production-ready
   4) Better to use official API with partnership

Q: How do you ensure data quality?
A: 1) Focus on verified accounts
   2) Filter by follower count threshold
   3) Engagement metric validation
   4) Ticker extraction and validation

Q: Alternative if LinkedIn API unavailable?
A: 1) RSS feeds from financial news sites
   2) Financial news APIs (Benzinga, Seeking Alpha)
   3) Press release aggregators (PR Newswire)
   4) SEC Edgar filings
"""
