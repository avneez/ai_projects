"""
Telegram Channel Real-time Monitor
===================================

This module implements a real-time Telegram channel monitor using Telethon.
It subscribes to financial news channels and extracts trading signals.

Key Features:
- Async event-driven architecture
- Multi-channel monitoring with single session
- Message filtering and ticker extraction
- Forwarded message tracking (source attribution)
- Media handling (images, documents)
- Rate limiting compliance

Author: TradeBrains Team
"""

import asyncio
import logging
import re
import hashlib
from datetime import datetime
from typing import List, Dict, Optional, Set
from dataclasses import dataclass, asdict
import json

from telethon import TelegramClient, events
from telethon.tl.types import MessageMediaPhoto, MessageMediaDocument
import redis
from kafka import KafkaProducer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TelegramMessage:
    """Structured Telegram message data model"""
    message_id: int
    channel_id: int
    channel_username: str
    text: str
    date: str
    sender_id: Optional[int]
    tickers: List[str]
    views: int
    forwards: int
    has_media: bool
    media_type: Optional[str]
    is_forwarded: bool
    forward_from: Optional[str]
    content_hash: str
    source: str = "telegram"


class TelegramChannelMonitor:
    """
    High-performance Telegram channel monitor for financial news.

    Architecture Decision:
    - Uses Telethon (async library) instead of python-telegram-bot
    - Single session manages multiple channel subscriptions
    - Event-driven architecture (push, not poll)
    - Redis for deduplication across channels

    Why Telethon?
    - Full MTProto API support (not just Bot API)
    - Can join channels without being admin
    - Async by default (better performance)
    - Access to message history and metadata
    """

    def __init__(
        self,
        api_id: int,
        api_hash: str,
        phone: str,
        session_name: str = "trading_bot",
        redis_host: str = "localhost",
        redis_port: int = 6379,
        kafka_bootstrap_servers: str = "localhost:9092"
    ):
        """
        Initialize Telegram monitor.

        Args:
            api_id: Telegram API ID (from my.telegram.org)
            api_hash: Telegram API Hash
            phone: Phone number for authentication
            session_name: Session file name (persists login)
            redis_host: Redis server for deduplication
            redis_port: Redis port
            kafka_bootstrap_servers: Kafka broker addresses
        """
        self.api_id = api_id
        self.api_hash = api_hash
        self.phone = phone
        self.session_name = session_name

        # Initialize Telegram client
        # Why session file? Avoids re-authentication on restart
        self.client = TelegramClient(session_name, api_id, api_hash)

        # Redis for deduplication
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=0,
            decode_responses=True,
            max_connections=50
        )

        # Kafka producer
        self.kafka_producer = KafkaProducer(
            bootstrap_servers=kafka_bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            compression_type='gzip',
            acks='all',
            retries=3
        )

        # Stock ticker patterns
        self.ticker_pattern = re.compile(r'\$[A-Z]{1,5}\b|[A-Z]{2,5}(?=\s|$)')

        # Financial keywords
        self.financial_keywords = [
            'breakout', 'resistance', 'support', 'rally', 'dump',
            'buy', 'sell', 'long', 'short', 'bullish', 'bearish',
            'earnings', 'revenue', 'profit', 'loss', 'dividend',
            'merger', 'acquisition', 'IPO', 'FDA', 'approval'
        ]

        # Monitored channels (will be populated)
        self.monitored_channels: Set[str] = set()

        logger.info("TelegramChannelMonitor initialized")

    def _extract_tickers(self, text: str) -> List[str]:
        """
        Extract stock tickers from message text.

        Handles multiple formats:
        - $AAPL (cashtag)
        - AAPL (plain ticker)
        - AAPL: (with colon)

        Args:
            text: Message text

        Returns:
            List of unique tickers
        """
        # Find all potential tickers
        potential_tickers = self.ticker_pattern.findall(text.upper())

        # Filter out common false positives
        false_positives = {'USD', 'USA', 'CEO', 'SEC', 'FDA', 'IPO', 'ETF', 'PM', 'AM'}
        tickers = [
            t.replace('$', '') for t in potential_tickers
            if t.replace('$', '') not in false_positives
        ]

        return list(set(tickers))

    def _compute_content_hash(self, text: str, channel_id: int) -> str:
        """
        Generate content hash for deduplication.

        Include channel_id to allow same content from different sources.

        Args:
            text: Message text
            channel_id: Telegram channel ID

        Returns:
            SHA256 hex digest
        """
        normalized = re.sub(r'http\S+|\s+', ' ', text.lower()).strip()
        content = f"{channel_id}:{normalized}"
        return hashlib.sha256(content.encode()).hexdigest()

    def _is_duplicate(self, content_hash: str) -> bool:
        """
        Check if message is duplicate using Redis.

        Args:
            content_hash: SHA256 hash

        Returns:
            True if duplicate
        """
        key = f"telegram:hash:{content_hash}"
        is_new = self.redis_client.set(key, "1", nx=True, ex=30*24*60*60)
        return not is_new

    def _has_financial_content(self, text: str) -> bool:
        """Check if message contains financial keywords."""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.financial_keywords)

    async def process_message(self, event) -> Optional[TelegramMessage]:
        """
        Process and validate Telegram message.

        Validation Steps:
        1. Extract text (handle media captions)
        2. Extract tickers
        3. Check financial relevance
        4. Deduplicate
        5. Structure data

        Args:
            event: Telethon NewMessage event

        Returns:
            Structured TelegramMessage or None
        """
        try:
            message = event.message

            # Extract text (include media caption if present)
            text = message.text or message.message or ""
            if not text and message.media:
                text = getattr(message.media, 'caption', '') or ""

            if not text:
                return None

            # Extract tickers
            tickers = self._extract_tickers(text)

            # Filter: Must have tickers OR financial keywords
            if not tickers and not self._has_financial_content(text):
                return None

            # Get channel information
            channel = await event.get_chat()
            channel_username = getattr(channel, 'username', 'unknown')

            # Handle forwarded messages
            is_forwarded = message.forward is not None
            forward_from = None
            if is_forwarded and message.forward.from_id:
                try:
                    forward_entity = await self.client.get_entity(message.forward.from_id)
                    forward_from = getattr(forward_entity, 'username', str(message.forward.from_id))
                except:
                    forward_from = "unknown"

            # Media handling
            has_media = message.media is not None
            media_type = None
            if has_media:
                if isinstance(message.media, MessageMediaPhoto):
                    media_type = "photo"
                elif isinstance(message.media, MessageMediaDocument):
                    media_type = "document"
                else:
                    media_type = "other"

            # Compute hash
            content_hash = self._compute_content_hash(text, channel.id)

            # Check duplicates
            if self._is_duplicate(content_hash):
                logger.debug(f"Message {message.id} is duplicate")
                return None

            # Structure message
            telegram_msg = TelegramMessage(
                message_id=message.id,
                channel_id=channel.id,
                channel_username=channel_username,
                text=text,
                date=message.date.isoformat(),
                sender_id=message.sender_id,
                tickers=tickers,
                views=message.views or 0,
                forwards=message.forwards or 0,
                has_media=has_media,
                media_type=media_type,
                is_forwarded=is_forwarded,
                forward_from=forward_from,
                content_hash=content_hash
            )

            logger.info(
                f"Processed message from @{channel_username} "
                f"with tickers: {telegram_msg.tickers}"
            )
            return telegram_msg

        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            return None

    def publish_to_kafka(self, message: TelegramMessage):
        """
        Publish message to Kafka topic.

        Args:
            message: Structured TelegramMessage
        """
        try:
            self.kafka_producer.send(
                topic='raw_telegram',
                value=asdict(message),
                key=str(message.message_id).encode('utf-8')
            )
            logger.debug(f"Published message {message.message_id} to Kafka")

        except Exception as e:
            logger.error(f"Failed to publish to Kafka: {e}", exc_info=True)

    async def add_channel(self, channel_username: str):
        """
        Add a channel to monitoring list.

        Args:
            channel_username: Telegram channel username (with or without @)
        """
        try:
            # Normalize username
            if not channel_username.startswith('@'):
                channel_username = f"@{channel_username}"

            # Check if already monitoring
            if channel_username in self.monitored_channels:
                logger.info(f"Already monitoring {channel_username}")
                return

            # Verify channel exists and we can access it
            channel = await self.client.get_entity(channel_username)
            logger.info(f"Successfully added channel: {channel_username} (ID: {channel.id})")

            self.monitored_channels.add(channel_username)

        except Exception as e:
            logger.error(f"Failed to add channel {channel_username}: {e}", exc_info=True)

    async def start_monitoring(self, channels: List[str]):
        """
        Start monitoring Telegram channels.

        Architecture:
        - Single client connection for all channels
        - Event-driven (no polling)
        - Async processing (non-blocking)

        Args:
            channels: List of channel usernames to monitor
        """
        # Start client
        await self.client.start(phone=self.phone)
        logger.info("Telegram client started and authenticated")

        # Add all channels
        for channel in channels:
            await self.add_channel(channel)

        if not self.monitored_channels:
            logger.warning("No channels to monitor!")
            return

        # Register event handler for new messages
        @self.client.on(events.NewMessage(chats=list(self.monitored_channels)))
        async def handler(event):
            """Event handler for new messages in monitored channels"""
            processed = await self.process_message(event)
            if processed:
                self.publish_to_kafka(processed)

        logger.info(f"Monitoring {len(self.monitored_channels)} channels: {self.monitored_channels}")
        logger.info("Press Ctrl+C to stop...")

        # Keep client running
        await self.client.run_until_disconnected()

    async def fetch_history(self, channel_username: str, limit: int = 100):
        """
        Fetch historical messages from a channel.

        Useful for:
        - Initial data backfill
        - Training news validation model
        - Historical sentiment analysis

        Args:
            channel_username: Channel username
            limit: Number of messages to fetch
        """
        try:
            channel = await self.client.get_entity(channel_username)
            logger.info(f"Fetching {limit} messages from {channel_username}")

            count = 0
            async for message in self.client.iter_messages(channel, limit=limit):
                # Create mock event for processing
                class MockEvent:
                    def __init__(self, msg, client):
                        self.message = msg
                        self.client = client

                    async def get_chat(self):
                        return channel

                event = MockEvent(message, self.client)
                processed = await self.process_message(event)

                if processed:
                    self.publish_to_kafka(processed)
                    count += 1

                # Rate limiting (Telegram allows ~30 req/sec)
                await asyncio.sleep(0.05)

            logger.info(f"Fetched {count} valid messages from {channel_username}")

        except Exception as e:
            logger.error(f"Failed to fetch history: {e}", exc_info=True)


# Example usage
if __name__ == "__main__":
    """
    Production deployment example.

    Environment Variables Required:
    - TELEGRAM_API_ID
    - TELEGRAM_API_HASH
    - TELEGRAM_PHONE
    - REDIS_HOST
    - KAFKA_BOOTSTRAP_SERVERS
    """
    import os

    # Configuration
    API_ID = int(os.getenv("TELEGRAM_API_ID"))
    API_HASH = os.getenv("TELEGRAM_API_HASH")
    PHONE = os.getenv("TELEGRAM_PHONE")  # e.g., +1234567890
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    KAFKA_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")

    if not all([API_ID, API_HASH, PHONE]):
        raise ValueError("Telegram credentials required")

    # Initialize monitor
    monitor = TelegramChannelMonitor(
        api_id=API_ID,
        api_hash=API_HASH,
        phone=PHONE,
        redis_host=REDIS_HOST,
        kafka_bootstrap_servers=KAFKA_SERVERS
    )

    # List of financial Telegram channels to monitor
    # These are examples - replace with actual channels
    CHANNELS_TO_MONITOR = [
        "@wallstreetbets",
        "@stockmarketlive",
        "@cryptonews",
        "@financeinsider",
        "@tradingtips"
    ]

    # Start monitoring
    try:
        asyncio.run(monitor.start_monitoring(CHANNELS_TO_MONITOR))
    except KeyboardInterrupt:
        logger.info("Monitoring stopped by user")
    except Exception as e:
        logger.error(f"Monitor failed: {e}", exc_info=True)


"""
INTERVIEW PREPARATION NOTES
===========================

Q: Why Telethon over python-telegram-bot?
A: 1) Telethon uses full MTProto API (not just Bot API)
   2) Can join channels without admin rights
   3) Async by default (better performance)
   4) Access to message history and metadata

Q: How do you handle Telegram rate limits?
A: 1) Telethon automatically handles flood wait
   2) Single session for multiple channels (connection pooling)
   3) Async processing prevents blocking
   4) ~30 messages/sec limit respected with sleep()

Q: What if a channel goes private?
A: 1) Monitor logs will show access error
   2) Implement retry logic with exponential backoff
   3) Alert monitoring system
   4) Keep monitoring other channels

Q: How do you ensure message ordering?
A: 1) Message ID is sequential per channel
   2) Timestamp in message metadata
   3) Kafka partitioning maintains order per channel

Q: What about forwarded messages?
A: Track original source in 'forward_from' field. Helps identify
   influential sources and prevent duplicate content from being treated
   as original.

Q: How do you scale for 100+ channels?
A: 1) Single client can handle 100+ channels efficiently
   2) For 1000+, use multiple clients with different sessions
   3) Partition channels across instances
   4) Kafka handles downstream scaling

Q: What happens if connection drops?
A: 1) Telethon auto-reconnects with exponential backoff
   2) Session file persists authentication
   3) No messages lost (can fetch missed via history)
   4) Event handlers automatically re-register

Q: How do you handle media (images, documents)?
A: 1) Track media presence and type in metadata
   2) Can download media using message.download_media()
   3) Store in object storage (S3/MinIO)
   4) OCR for images with text (optional)
"""
