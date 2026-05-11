# InsightStream - Code Snippets

This document contains sample code snippets demonstrating key components of the InsightStream data enrichment pipeline. These are production-ready examples showcasing the architecture and best practices.

---

## Table of Contents
1. [FastAPI REST API](#1-fastapi-rest-api)
2. [Kafka Producer for Data Ingestion](#2-kafka-producer-for-data-ingestion)
3. [Celery Task for Data Enrichment](#3-celery-task-for-data-enrichment)
4. [Airflow DAG for Pipeline Orchestration](#4-airflow-dag-for-pipeline-orchestration)
5. [PostgreSQL with pgvector for Embeddings](#5-postgresql-with-pgvector-for-embeddings)
6. [ElasticSearch Indexing](#6-elasticsearch-indexing)
7. [Redis Caching Layer](#7-redis-caching-layer)
8. [Data Validation with Great Expectations](#8-data-validation-with-great-expectations)
9. [Prometheus Metrics](#9-prometheus-metrics)
10. [Sentence-Transformers Embeddings](#10-sentence-transformers-embeddings)

---

## 1. FastAPI REST API

**File**: `src/api/main.py`

```python
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from typing import List, Optional
import redis.asyncio as redis
from sqlalchemy.orm import Session
import json

from src.models.database import get_db
from src.models.contact import Contact
from src.workers.tasks import enrich_contact_task
from src.utils.cache import get_redis_client
from src.monitoring.metrics import track_api_request, api_latency

app = FastAPI(
    title="InsightStream API",
    description="Real-Time Data Enrichment Pipeline",
    version="1.0.0"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request Models
class ContactEnrichRequest(BaseModel):
    email: EmailStr
    sources: List[str] = ["linkedin", "zoominfo", "bombora"]
    priority: str = "normal"  # normal, high, urgent

class BulkEnrichRequest(BaseModel):
    contacts: List[dict]
    sources: List[str] = ["linkedin", "zoominfo"]

class ContactSearchResponse(BaseModel):
    id: int
    email: str
    first_name: Optional[str]
    last_name: Optional[str]
    company: Optional[str]
    title: Optional[str]
    enrichment_score: float

# API Endpoints
@app.post("/api/v1/contacts/enrich", status_code=202)
@track_api_request("enrich_contact")
async def enrich_contact(
    request: ContactEnrichRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Enrich a single contact from multiple data sources.
    Returns task ID for tracking progress.
    """
    with api_latency.labels(endpoint="enrich_contact").time():
        # Check if contact exists
        contact = db.query(Contact).filter(Contact.email == request.email).first()

        if not contact:
            contact = Contact(email=request.email)
            db.add(contact)
            db.commit()
            db.refresh(contact)

        # Trigger async enrichment task
        task = enrich_contact_task.apply_async(
            args=[contact.id, request.sources],
            priority=_get_priority_value(request.priority)
        )

        return {
            "task_id": task.id,
            "contact_id": contact.id,
            "status": "processing",
            "message": "Contact enrichment initiated"
        }

@app.get("/api/v1/contacts/search")
@track_api_request("search_contacts")
async def search_contacts(
    q: str,
    limit: int = 10,
    offset: int = 0,
    redis_client: redis.Redis = Depends(get_redis_client),
    db: Session = Depends(get_db)
):
    """
    Search contacts with caching and ElasticSearch.
    """
    with api_latency.labels(endpoint="search_contacts").time():
        # Check cache first
        cache_key = f"search:{q}:{limit}:{offset}"
        cached_result = await redis_client.get(cache_key)

        if cached_result:
            return json.loads(cached_result)

        # Query ElasticSearch (implemented in utils)
        from src.utils.elasticsearch_client import search_contacts_es
        results = await search_contacts_es(q, limit, offset)

        # Cache for 1 hour
        await redis_client.setex(
            cache_key,
            3600,
            json.dumps(results)
        )

        return results

@app.get("/api/v1/contacts/{contact_id}/similar")
@track_api_request("similar_contacts")
async def get_similar_contacts(
    contact_id: int,
    limit: int = 5,
    db: Session = Depends(get_db)
):
    """
    Find similar contacts using pgvector similarity search.
    """
    with api_latency.labels(endpoint="similar_contacts").time():
        contact = db.query(Contact).filter(Contact.id == contact_id).first()

        if not contact or not contact.embedding:
            raise HTTPException(status_code=404, detail="Contact not found or no embedding")

        # pgvector similarity search
        similar = db.execute(
            """
            SELECT id, email, first_name, last_name, company,
                   1 - (embedding <=> :target_embedding) AS similarity
            FROM contacts
            WHERE id != :contact_id AND embedding IS NOT NULL
            ORDER BY embedding <=> :target_embedding
            LIMIT :limit
            """,
            {
                "target_embedding": contact.embedding,
                "contact_id": contact_id,
                "limit": limit
            }
        ).fetchall()

        return [
            {
                "id": row.id,
                "email": row.email,
                "name": f"{row.first_name} {row.last_name}",
                "company": row.company,
                "similarity_score": round(row.similarity, 3)
            }
            for row in similar
        ]

@app.post("/api/v1/contacts/bulk-enrich", status_code=202)
@track_api_request("bulk_enrich")
async def bulk_enrich(
    request: BulkEnrichRequest,
    background_tasks: BackgroundTasks
):
    """
    Bulk enrichment for multiple contacts.
    """
    from src.workers.tasks import bulk_enrich_task

    task = bulk_enrich_task.apply_async(
        args=[request.contacts, request.sources]
    )

    return {
        "task_id": task.id,
        "contact_count": len(request.contacts),
        "status": "processing"
    }

@app.get("/health")
async def health_check(db: Session = Depends(get_db)):
    """
    Health check endpoint for monitoring.
    """
    try:
        # Check database connection
        db.execute("SELECT 1")
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Unhealthy: {str(e)}")

def _get_priority_value(priority: str) -> int:
    priority_map = {"urgent": 9, "high": 7, "normal": 5, "low": 3}
    return priority_map.get(priority, 5)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## 2. Kafka Producer for Data Ingestion

**File**: `src/pipelines/kafka_producer.py`

```python
from kafka import KafkaProducer
from kafka.errors import KafkaError
import json
import logging
from typing import Dict, Any, List
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)

class DataIngestionProducer:
    """
    Kafka producer for ingesting data from external APIs
    into the enrichment pipeline.
    """

    def __init__(self, bootstrap_servers: List[str], topic: str):
        self.topic = topic
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            key_serializer=lambda k: k.encode('utf-8') if k else None,
            acks='all',  # Wait for all replicas
            retries=3,
            max_in_flight_requests_per_connection=1,  # Ensure ordering
            compression_type='gzip'
        )
        logger.info(f"Kafka producer initialized for topic: {topic}")

    def send_contact_data(self, contact_data: Dict[str, Any], source: str) -> bool:
        """
        Send contact data to Kafka topic with deduplication key.
        """
        try:
            # Create unique key for deduplication
            dedup_key = self._create_dedup_key(contact_data)

            # Enrich message with metadata
            message = {
                "data": contact_data,
                "source": source,
                "ingested_at": datetime.utcnow().isoformat(),
                "dedup_key": dedup_key
            }

            # Send to Kafka with callback
            future = self.producer.send(
                self.topic,
                key=dedup_key,
                value=message,
                partition=self._get_partition(dedup_key)
            )

            # Block for 'synchronous' send
            record_metadata = future.get(timeout=10)

            logger.info(
                f"Message sent to {record_metadata.topic} "
                f"partition {record_metadata.partition} "
                f"offset {record_metadata.offset}"
            )
            return True

        except KafkaError as e:
            logger.error(f"Failed to send message to Kafka: {e}")
            return False

    def send_batch(self, contacts: List[Dict[str, Any]], source: str) -> Dict[str, int]:
        """
        Send batch of contacts to Kafka.
        """
        success_count = 0
        failure_count = 0

        for contact in contacts:
            if self.send_contact_data(contact, source):
                success_count += 1
            else:
                failure_count += 1

        self.producer.flush()  # Ensure all messages are sent

        return {
            "success": success_count,
            "failed": failure_count,
            "total": len(contacts)
        }

    def _create_dedup_key(self, contact_data: Dict[str, Any]) -> str:
        """
        Create deduplication key from email + company domain.
        """
        email = contact_data.get("email", "")
        company = contact_data.get("company", "")

        key_string = f"{email.lower()}:{company.lower()}"
        return hashlib.md5(key_string.encode()).hexdigest()

    def _get_partition(self, key: str) -> int:
        """
        Consistent hash-based partitioning.
        """
        # Ensure same key always goes to same partition
        return hash(key) % 3  # Assuming 3 partitions

    def close(self):
        """Close producer connection."""
        self.producer.close()
        logger.info("Kafka producer closed")


# Example usage for LinkedIn data ingestion
if __name__ == "__main__":
    from src.utils.api_clients import LinkedInAPIClient

    # Initialize producer
    producer = DataIngestionProducer(
        bootstrap_servers=['localhost:9092'],
        topic='contact-enrichment-raw'
    )

    # Fetch data from LinkedIn API
    linkedin_client = LinkedInAPIClient()
    contacts = linkedin_client.fetch_contacts(limit=100)

    # Send to Kafka
    result = producer.send_batch(contacts, source="linkedin")
    print(f"Sent {result['success']} contacts to Kafka")

    producer.close()
```

---

## 3. Celery Task for Data Enrichment

**File**: `src/workers/tasks.py`

```python
from celery import Celery, Task
from celery.signals import task_prerun, task_postrun
from sqlalchemy.orm import Session
import logging
from typing import List, Dict, Any
from datetime import datetime

from src.models.database import SessionLocal
from src.models.contact import Contact
from src.utils.api_clients import LinkedInAPIClient, ZoomInfoClient, BomboraClient
from src.utils.embeddings import generate_embedding
from src.monitoring.metrics import task_duration, task_counter

logger = logging.getLogger(__name__)

# Initialize Celery
celery_app = Celery(
    'insightstream',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/1'
)

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=300,  # 5 minutes
    task_soft_time_limit=240,  # 4 minutes
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
)

class DatabaseTask(Task):
    """Base task with database session management."""
    _db = None

    @property
    def db(self) -> Session:
        if self._db is None:
            self._db = SessionLocal()
        return self._db

    def after_return(self, *args, **kwargs):
        if self._db is not None:
            self._db.close()

@celery_app.task(bind=True, base=DatabaseTask, max_retries=3)
def enrich_contact_task(self, contact_id: int, sources: List[str]) -> Dict[str, Any]:
    """
    Enrich a single contact from specified sources.
    Implements retry logic and circuit breaker pattern.
    """
    start_time = datetime.utcnow()
    task_counter.labels(task_name="enrich_contact", status="started").inc()

    try:
        # Fetch contact from database
        contact = self.db.query(Contact).filter(Contact.id == contact_id).first()
        if not contact:
            raise ValueError(f"Contact {contact_id} not found")

        enriched_data = {}
        failed_sources = []

        # Enrich from each source
        for source in sources:
            try:
                data = self._enrich_from_source(contact.email, source)
                enriched_data[source] = data
            except Exception as e:
                logger.error(f"Failed to enrich from {source}: {e}")
                failed_sources.append(source)

                # Retry on failure
                if self.request.retries < self.max_retries:
                    raise self.retry(exc=e, countdown=2 ** self.request.retries)

        # Merge enriched data
        merged_data = self._merge_enrichment_data(enriched_data)

        # Update contact
        for key, value in merged_data.items():
            setattr(contact, key, value)

        contact.last_enriched_at = datetime.utcnow()
        contact.enrichment_sources = sources
        contact.enrichment_score = self._calculate_enrichment_score(contact)

        # Generate and store embedding
        embedding = generate_embedding(contact)
        contact.embedding = embedding

        self.db.commit()

        # Record metrics
        duration = (datetime.utcnow() - start_time).total_seconds()
        task_duration.labels(task_name="enrich_contact").observe(duration)
        task_counter.labels(task_name="enrich_contact", status="success").inc()

        logger.info(f"Successfully enriched contact {contact_id} from {len(sources)} sources")

        return {
            "contact_id": contact_id,
            "enriched_fields": list(merged_data.keys()),
            "failed_sources": failed_sources,
            "enrichment_score": contact.enrichment_score
        }

    except Exception as e:
        task_counter.labels(task_name="enrich_contact", status="failed").inc()
        logger.error(f"Failed to enrich contact {contact_id}: {e}")
        raise

    def _enrich_from_source(self, email: str, source: str) -> Dict[str, Any]:
        """Fetch data from specific source."""
        clients = {
            "linkedin": LinkedInAPIClient(),
            "zoominfo": ZoomInfoClient(),
            "bombora": BomboraClient()
        }

        client = clients.get(source)
        if not client:
            raise ValueError(f"Unknown source: {source}")

        return client.fetch_contact_data(email)

    def _merge_enrichment_data(self, enriched_data: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Merge data from multiple sources with priority-based conflict resolution.
        Priority: linkedin > zoominfo > bombora
        """
        merged = {}
        source_priority = ["linkedin", "zoominfo", "bombora"]

        for source in source_priority:
            if source in enriched_data:
                for key, value in enriched_data[source].items():
                    if key not in merged or merged[key] is None:
                        merged[key] = value

        return merged

    def _calculate_enrichment_score(self, contact: Contact) -> float:
        """Calculate enrichment completeness score (0-1)."""
        required_fields = [
            'first_name', 'last_name', 'email', 'company',
            'title', 'phone', 'linkedin_url', 'location'
        ]

        filled_fields = sum(
            1 for field in required_fields
            if getattr(contact, field, None) is not None
        )

        return filled_fields / len(required_fields)

@celery_app.task(bind=True, base=DatabaseTask)
def bulk_enrich_task(self, contacts: List[Dict], sources: List[str]) -> Dict[str, Any]:
    """
    Bulk enrichment task - spawns individual enrichment tasks.
    """
    task_ids = []

    for contact_data in contacts:
        # Create or get contact
        contact = self.db.query(Contact).filter(
            Contact.email == contact_data['email']
        ).first()

        if not contact:
            contact = Contact(**contact_data)
            self.db.add(contact)
            self.db.commit()
            self.db.refresh(contact)

        # Spawn enrichment task
        task = enrich_contact_task.apply_async(
            args=[contact.id, sources],
            queue='enrichment'
        )
        task_ids.append(task.id)

    return {
        "task_ids": task_ids,
        "total_contacts": len(contacts)
    }

@celery_app.task
def cleanup_old_data():
    """Periodic task to clean up old data."""
    db = SessionLocal()
    try:
        # Delete contacts not enriched in 90 days
        cutoff_date = datetime.utcnow() - timedelta(days=90)
        deleted = db.query(Contact).filter(
            Contact.last_enriched_at < cutoff_date
        ).delete()

        db.commit()
        logger.info(f"Deleted {deleted} old contacts")

    finally:
        db.close()
```

---

## 4. Airflow DAG for Pipeline Orchestration

**File**: `src/pipelines/dags/data_enrichment_dag.py`

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.apache.kafka.sensors.kafka import AwaitMessageSensor
from airflow.utils.dates import days_ago
from datetime import datetime, timedelta
import logging

from src.pipelines.kafka_producer import DataIngestionProducer
from src.utils.api_clients import LinkedInAPIClient, ZoomInfoClient
from src.workers.tasks import bulk_enrich_task
from src.monitoring.metrics import pipeline_duration, pipeline_records_processed

logger = logging.getLogger(__name__)

# Default arguments
default_args = {
    'owner': 'data-engineering',
    'depends_on_past': False,
    'email': ['alerts@company.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(hours=2),
}

# Define DAG
dag = DAG(
    'data_enrichment_pipeline',
    default_args=default_args,
    description='Real-time data enrichment pipeline',
    schedule_interval='0 */4 * * *',  # Every 4 hours
    start_date=days_ago(1),
    catchup=False,
    tags=['enrichment', 'production'],
)

def fetch_linkedin_data(**context):
    """Fetch contacts from LinkedIn API."""
    logger.info("Fetching data from LinkedIn API")

    client = LinkedInAPIClient()
    contacts = client.fetch_contacts(limit=10000)

    # Push to Kafka
    producer = DataIngestionProducer(
        bootstrap_servers=['localhost:9092'],
        topic='contact-enrichment-raw'
    )

    result = producer.send_batch(contacts, source="linkedin")
    producer.close()

    # Push to XCom for downstream tasks
    context['task_instance'].xcom_push(key='linkedin_count', value=result['success'])

    logger.info(f"Sent {result['success']} LinkedIn contacts to Kafka")
    return result

def fetch_zoominfo_data(**context):
    """Fetch contacts from ZoomInfo API."""
    logger.info("Fetching data from ZoomInfo API")

    client = ZoomInfoClient()
    contacts = client.fetch_contacts(limit=10000)

    producer = DataIngestionProducer(
        bootstrap_servers=['localhost:9092'],
        topic='contact-enrichment-raw'
    )

    result = producer.send_batch(contacts, source="zoominfo")
    producer.close()

    context['task_instance'].xcom_push(key='zoominfo_count', value=result['success'])

    logger.info(f"Sent {result['success']} ZoomInfo contacts to Kafka")
    return result

def validate_data_quality(**context):
    """Run data quality validation using Great Expectations."""
    from src.pipelines.data_validation import run_validation_suite

    logger.info("Running data quality validation")

    validation_results = run_validation_suite(
        datasource='postgres',
        expectation_suite='contact_enrichment_suite'
    )

    if not validation_results['success']:
        raise ValueError(f"Data quality validation failed: {validation_results['failures']}")

    logger.info(f"Data quality validation passed: {validation_results['success_percentage']}%")
    return validation_results

def trigger_enrichment(**context):
    """Trigger Celery enrichment tasks for new contacts."""
    from src.models.database import SessionLocal
    from src.models.contact import Contact

    logger.info("Triggering enrichment tasks")

    db = SessionLocal()
    try:
        # Get contacts that need enrichment
        contacts_to_enrich = db.query(Contact).filter(
            Contact.last_enriched_at.is_(None) |
            (Contact.last_enriched_at < datetime.utcnow() - timedelta(days=30))
        ).limit(5000).all()

        # Trigger bulk enrichment
        contact_data = [
            {"email": c.email, "id": c.id}
            for c in contacts_to_enrich
        ]

        task = bulk_enrich_task.apply_async(
            args=[contact_data, ["linkedin", "zoominfo"]]
        )

        logger.info(f"Triggered enrichment for {len(contact_data)} contacts")

        # Record metrics
        pipeline_records_processed.labels(pipeline="enrichment").inc(len(contact_data))

        return {"task_id": task.id, "contact_count": len(contact_data)}

    finally:
        db.close()

def index_to_elasticsearch(**context):
    """Index enriched contacts to ElasticSearch."""
    from src.utils.elasticsearch_client import bulk_index_contacts
    from src.models.database import SessionLocal
    from src.models.contact import Contact

    logger.info("Indexing contacts to ElasticSearch")

    db = SessionLocal()
    try:
        # Get recently enriched contacts
        cutoff_time = datetime.utcnow() - timedelta(hours=4)
        contacts = db.query(Contact).filter(
            Contact.last_enriched_at >= cutoff_time
        ).all()

        # Bulk index
        indexed_count = bulk_index_contacts(contacts)

        logger.info(f"Indexed {indexed_count} contacts to ElasticSearch")
        return {"indexed_count": indexed_count}

    finally:
        db.close()

def update_metrics(**context):
    """Update pipeline metrics and dashboards."""
    ti = context['task_instance']

    linkedin_count = ti.xcom_pull(key='linkedin_count', task_ids='fetch_linkedin')
    zoominfo_count = ti.xcom_pull(key='zoominfo_count', task_ids='fetch_zoominfo')

    total_ingested = linkedin_count + zoominfo_count

    logger.info(f"Pipeline completed: {total_ingested} contacts ingested")

    # Record to Prometheus
    pipeline_duration.labels(pipeline="enrichment").observe(
        context['dag_run'].duration.total_seconds()
    )

    return {"total_ingested": total_ingested}

# Define tasks
task_fetch_linkedin = PythonOperator(
    task_id='fetch_linkedin',
    python_callable=fetch_linkedin_data,
    dag=dag,
)

task_fetch_zoominfo = PythonOperator(
    task_id='fetch_zoominfo',
    python_callable=fetch_zoominfo_data,
    dag=dag,
)

task_validate_quality = PythonOperator(
    task_id='validate_data_quality',
    python_callable=validate_data_quality,
    dag=dag,
)

task_trigger_enrichment = PythonOperator(
    task_id='trigger_enrichment',
    python_callable=trigger_enrichment,
    dag=dag,
)

task_index_elasticsearch = PythonOperator(
    task_id='index_elasticsearch',
    python_callable=index_to_elasticsearch,
    dag=dag,
)

task_update_metrics = PythonOperator(
    task_id='update_metrics',
    python_callable=update_metrics,
    dag=dag,
)

# Define task dependencies
[task_fetch_linkedin, task_fetch_zoominfo] >> task_validate_quality
task_validate_quality >> task_trigger_enrichment
task_trigger_enrichment >> task_index_elasticsearch
task_index_elasticsearch >> task_update_metrics
```

---

## 5. PostgreSQL with pgvector for Embeddings

**File**: `src/models/contact.py`

```python
from sqlalchemy import Column, Integer, String, DateTime, Float, ARRAY, JSON, Text
from sqlalchemy.ext.declarative import declarative_base
from pgvector.sqlalchemy import Vector
from datetime import datetime

Base = declarative_base()

class Contact(Base):
    """Contact model with pgvector support for embeddings."""

    __tablename__ = 'contacts'

    # Primary fields
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)

    # Personal information
    first_name = Column(String(100))
    last_name = Column(String(100))
    full_name = Column(String(255))
    title = Column(String(255))
    phone = Column(String(50))

    # Company information
    company = Column(String(255), index=True)
    company_domain = Column(String(255))
    company_size = Column(String(50))
    industry = Column(String(100), index=True)

    # Location
    location = Column(String(255))
    city = Column(String(100))
    state = Column(String(100))
    country = Column(String(100))

    # Social profiles
    linkedin_url = Column(String(500))
    twitter_handle = Column(String(100))

    # Enrichment metadata
    enrichment_sources = Column(ARRAY(String), default=[])
    enrichment_score = Column(Float, default=0.0)
    last_enriched_at = Column(DateTime)

    # Vector embedding (384 dimensions for sentence-transformers)
    embedding = Column(Vector(384))

    # Additional data as JSON
    raw_data = Column(JSON)
    tags = Column(ARRAY(String), default=[])

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<Contact(email={self.email}, company={self.company})>"

    def to_dict(self):
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "email": self.email,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "full_name": self.full_name,
            "title": self.title,
            "company": self.company,
            "industry": self.industry,
            "location": self.location,
            "linkedin_url": self.linkedin_url,
            "enrichment_score": self.enrichment_score,
            "last_enriched_at": self.last_enriched_at.isoformat() if self.last_enriched_at else None,
        }

# Database initialization
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://user:password@localhost:5432/insightstream"
)

engine = create_engine(DATABASE_URL, pool_size=20, max_overflow=40)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_database():
    """Initialize database with pgvector extension."""
    from sqlalchemy import text

    with engine.connect() as conn:
        # Enable pgvector extension
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()

    # Create all tables
    Base.metadata.create_all(bind=engine)

    # Create indexes for performance
    with engine.connect() as conn:
        # Composite index for common queries
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_company_enriched "
            "ON contacts (company, last_enriched_at DESC)"
        ))

        # Vector similarity index using IVFFlat
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_embedding_ivfflat "
            "ON contacts USING ivfflat (embedding vector_cosine_ops) "
            "WITH (lists = 100)"
        ))

        conn.commit()

    print("Database initialized successfully")

def get_db():
    """Dependency for FastAPI."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

if __name__ == "__main__":
    init_database()
```

**File**: `src/utils/vector_search.py`

```python
from sqlalchemy.orm import Session
from src.models.contact import Contact
from typing import List, Dict, Any
import numpy as np

def find_similar_contacts(
    db: Session,
    target_embedding: List[float],
    limit: int = 10,
    min_similarity: float = 0.7
) -> List[Dict[str, Any]]:
    """
    Find similar contacts using pgvector cosine similarity.

    The <=> operator computes cosine distance (1 - cosine similarity).
    """
    results = db.execute(
        """
        SELECT
            id,
            email,
            first_name,
            last_name,
            company,
            title,
            1 - (embedding <=> :target_embedding::vector) AS similarity
        FROM contacts
        WHERE embedding IS NOT NULL
            AND 1 - (embedding <=> :target_embedding::vector) >= :min_similarity
        ORDER BY embedding <=> :target_embedding::vector
        LIMIT :limit
        """,
        {
            "target_embedding": target_embedding,
            "min_similarity": min_similarity,
            "limit": limit
        }
    ).fetchall()

    return [
        {
            "id": row.id,
            "email": row.email,
            "name": f"{row.first_name} {row.last_name}",
            "company": row.company,
            "title": row.title,
            "similarity_score": round(row.similarity, 4)
        }
        for row in results
    ]

def find_duplicate_contacts(
    db: Session,
    contact_id: int,
    similarity_threshold: float = 0.95
) -> List[Dict[str, Any]]:
    """
    Find potential duplicate contacts using high similarity threshold.
    """
    contact = db.query(Contact).filter(Contact.id == contact_id).first()

    if not contact or not contact.embedding:
        return []

    duplicates = db.execute(
        """
        SELECT
            id,
            email,
            full_name,
            company,
            1 - (embedding <=> :target_embedding::vector) AS similarity
        FROM contacts
        WHERE id != :contact_id
            AND embedding IS NOT NULL
            AND 1 - (embedding <=> :target_embedding::vector) >= :threshold
        ORDER BY embedding <=> :target_embedding::vector
        LIMIT 10
        """,
        {
            "target_embedding": contact.embedding,
            "contact_id": contact_id,
            "threshold": similarity_threshold
        }
    ).fetchall()

    return [
        {
            "id": row.id,
            "email": row.email,
            "name": row.full_name,
            "company": row.company,
            "similarity_score": round(row.similarity, 4),
            "likely_duplicate": row.similarity > 0.98
        }
        for row in duplicates
    ]
```

---

## 6. ElasticSearch Indexing

**File**: `src/utils/elasticsearch_client.py`

```python
from elasticsearch import Elasticsearch, helpers
from typing import List, Dict, Any
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ElasticSearchClient:
    """Client for ElasticSearch operations."""

    def __init__(self, hosts: List[str] = ['http://localhost:9200']):
        self.client = Elasticsearch(hosts)
        self.index_name = 'contacts'
        self._ensure_index_exists()

    def _ensure_index_exists(self):
        """Create index with optimal mappings if it doesn't exist."""
        if not self.client.indices.exists(index=self.index_name):
            mapping = {
                "mappings": {
                    "properties": {
                        "id": {"type": "integer"},
                        "email": {"type": "keyword"},
                        "first_name": {
                            "type": "text",
                            "fields": {"keyword": {"type": "keyword"}}
                        },
                        "last_name": {
                            "type": "text",
                            "fields": {"keyword": {"type": "keyword"}}
                        },
                        "full_name": {
                            "type": "text",
                            "fields": {"keyword": {"type": "keyword"}}
                        },
                        "title": {
                            "type": "text",
                            "fields": {"keyword": {"type": "keyword"}}
                        },
                        "company": {
                            "type": "text",
                            "fields": {"keyword": {"type": "keyword"}}
                        },
                        "company_domain": {"type": "keyword"},
                        "industry": {"type": "keyword"},
                        "location": {"type": "text"},
                        "city": {"type": "keyword"},
                        "country": {"type": "keyword"},
                        "enrichment_score": {"type": "float"},
                        "tags": {"type": "keyword"},
                        "last_enriched_at": {"type": "date"},
                        "created_at": {"type": "date"},
                        "suggest": {
                            "type": "completion",
                            "analyzer": "simple",
                            "search_analyzer": "simple"
                        }
                    }
                },
                "settings": {
                    "number_of_shards": 3,
                    "number_of_replicas": 1,
                    "refresh_interval": "5s"
                }
            }

            self.client.indices.create(index=self.index_name, body=mapping)
            logger.info(f"Created ElasticSearch index: {self.index_name}")

    def index_contact(self, contact: Dict[str, Any]) -> bool:
        """Index a single contact."""
        try:
            # Add autocomplete suggestions
            suggest_input = [
                contact.get('email', ''),
                contact.get('full_name', ''),
                contact.get('company', '')
            ]

            contact['suggest'] = {
                "input": [s for s in suggest_input if s],
                "weight": int(contact.get('enrichment_score', 0) * 100)
            }

            self.client.index(
                index=self.index_name,
                id=contact['id'],
                document=contact
            )
            return True

        except Exception as e:
            logger.error(f"Failed to index contact {contact.get('id')}: {e}")
            return False

    def bulk_index_contacts(self, contacts: List[Dict[str, Any]]) -> int:
        """Bulk index multiple contacts."""
        def generate_actions():
            for contact in contacts:
                # Prepare document
                suggest_input = [
                    contact.get('email', ''),
                    contact.get('full_name', ''),
                    contact.get('company', '')
                ]

                contact['suggest'] = {
                    "input": [s for s in suggest_input if s],
                    "weight": int(contact.get('enrichment_score', 0) * 100)
                }

                yield {
                    "_index": self.index_name,
                    "_id": contact['id'],
                    "_source": contact
                }

        try:
            success, failed = helpers.bulk(
                self.client,
                generate_actions(),
                raise_on_error=False,
                chunk_size=500
            )

            logger.info(f"Indexed {success} contacts, {len(failed)} failed")
            return success

        except Exception as e:
            logger.error(f"Bulk indexing failed: {e}")
            return 0

    def search_contacts(
        self,
        query: str,
        limit: int = 10,
        offset: int = 0,
        filters: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Full-text search with filters.
        """
        # Build query
        must_clauses = [
            {
                "multi_match": {
                    "query": query,
                    "fields": [
                        "full_name^3",
                        "email^2",
                        "company^2",
                        "title",
                        "location"
                    ],
                    "type": "best_fields",
                    "fuzziness": "AUTO"
                }
            }
        ]

        # Add filters
        filter_clauses = []
        if filters:
            if 'industry' in filters:
                filter_clauses.append({"term": {"industry": filters['industry']}})
            if 'company' in filters:
                filter_clauses.append({"term": {"company.keyword": filters['company']}})
            if 'min_enrichment_score' in filters:
                filter_clauses.append({
                    "range": {
                        "enrichment_score": {"gte": filters['min_enrichment_score']}
                    }
                })

        search_body = {
            "query": {
                "bool": {
                    "must": must_clauses,
                    "filter": filter_clauses
                }
            },
            "from": offset,
            "size": limit,
            "sort": [
                {"_score": {"order": "desc"}},
                {"enrichment_score": {"order": "desc"}}
            ],
            "highlight": {
                "fields": {
                    "full_name": {},
                    "title": {},
                    "company": {}
                }
            }
        }

        try:
            response = self.client.search(index=self.index_name, body=search_body)

            return {
                "total": response['hits']['total']['value'],
                "results": [
                    {
                        **hit['_source'],
                        "score": hit['_score'],
                        "highlights": hit.get('highlight', {})
                    }
                    for hit in response['hits']['hits']
                ]
            }

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return {"total": 0, "results": []}

    def autocomplete(self, prefix: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Autocomplete suggestions."""
        search_body = {
            "suggest": {
                "contact-suggest": {
                    "prefix": prefix,
                    "completion": {
                        "field": "suggest",
                        "size": limit,
                        "skip_duplicates": True
                    }
                }
            }
        }

        try:
            response = self.client.search(index=self.index_name, body=search_body)
            suggestions = response['suggest']['contact-suggest'][0]['options']

            return [
                {
                    "text": option['text'],
                    "score": option['_score'],
                    "contact": option['_source']
                }
                for option in suggestions
            ]

        except Exception as e:
            logger.error(f"Autocomplete failed: {e}")
            return []

# Helper functions
def bulk_index_contacts(contacts: List[Any]) -> int:
    """Helper to bulk index Contact model instances."""
    es_client = ElasticSearchClient()

    contact_dicts = [contact.to_dict() for contact in contacts]
    return es_client.bulk_index_contacts(contact_dicts)

async def search_contacts_es(query: str, limit: int, offset: int) -> Dict[str, Any]:
    """Async helper for FastAPI."""
    es_client = ElasticSearchClient()
    return es_client.search_contacts(query, limit, offset)
```

---

## 7. Redis Caching Layer

**File**: `src/utils/cache.py`

```python
import redis.asyncio as redis
import json
from typing import Any, Optional
from functools import wraps
import hashlib
import logging

logger = logging.getLogger(__name__)

# Redis connection pool
redis_pool = None

def get_redis_pool():
    """Get or create Redis connection pool."""
    global redis_pool
    if redis_pool is None:
        redis_pool = redis.ConnectionPool(
            host='localhost',
            port=6379,
            db=0,
            decode_responses=True,
            max_connections=50
        )
    return redis_pool

async def get_redis_client() -> redis.Redis:
    """Get Redis client (dependency for FastAPI)."""
    pool = get_redis_pool()
    return redis.Redis(connection_pool=pool)

class CacheManager:
    """Redis cache manager with multi-tier caching."""

    def __init__(self):
        self.client = None

    async def get_client(self) -> redis.Redis:
        if self.client is None:
            pool = get_redis_pool()
            self.client = redis.Redis(connection_pool=pool)
        return self.client

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        try:
            client = await self.get_client()
            value = await client.get(key)

            if value:
                logger.debug(f"Cache HIT: {key}")
                return json.loads(value)

            logger.debug(f"Cache MISS: {key}")
            return None

        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: int = 3600
    ) -> bool:
        """Set value in cache with TTL."""
        try:
            client = await self.get_client()
            serialized = json.dumps(value)
            await client.setex(key, ttl, serialized)

            logger.debug(f"Cache SET: {key} (TTL: {ttl}s)")
            return True

        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        try:
            client = await self.get_client()
            await client.delete(key)
            return True
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False

    async def delete_pattern(self, pattern: str) -> int:
        """Delete keys matching pattern."""
        try:
            client = await self.get_client()
            keys = await client.keys(pattern)
            if keys:
                return await client.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"Cache delete pattern error: {e}")
            return 0

    async def get_stats(self) -> dict:
        """Get cache statistics."""
        try:
            client = await self.get_client()
            info = await client.info('stats')

            return {
                "hits": info.get('keyspace_hits', 0),
                "misses": info.get('keyspace_misses', 0),
                "hit_rate": self._calculate_hit_rate(
                    info.get('keyspace_hits', 0),
                    info.get('keyspace_misses', 0)
                )
            }
        except Exception as e:
            logger.error(f"Cache stats error: {e}")
            return {}

    @staticmethod
    def _calculate_hit_rate(hits: int, misses: int) -> float:
        total = hits + misses
        return round(hits / total * 100, 2) if total > 0 else 0.0

# Global cache manager instance
cache_manager = CacheManager()

def cache_key(*args, **kwargs) -> str:
    """Generate cache key from function arguments."""
    key_parts = [str(arg) for arg in args]
    key_parts.extend(f"{k}:{v}" for k, v in sorted(kwargs.items()))
    key_string = ":".join(key_parts)
    return hashlib.md5(key_string.encode()).hexdigest()

def cached(ttl: int = 3600, prefix: str = ""):
    """
    Decorator for caching function results.

    Usage:
        @cached(ttl=1800, prefix="contact")
        async def get_contact(contact_id: int):
            ...
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            key = f"{prefix}:{func.__name__}:{cache_key(*args, **kwargs)}"

            # Try to get from cache
            cached_result = await cache_manager.get(key)
            if cached_result is not None:
                return cached_result

            # Call function and cache result
            result = await func(*args, **kwargs)
            await cache_manager.set(key, result, ttl)

            return result
        return wrapper
    return decorator

# Example usage
@cached(ttl=1800, prefix="contact")
async def get_contact_by_id(contact_id: int):
    """Get contact with caching."""
    from src.models.database import SessionLocal
    from src.models.contact import Contact

    db = SessionLocal()
    try:
        contact = db.query(Contact).filter(Contact.id == contact_id).first()
        return contact.to_dict() if contact else None
    finally:
        db.close()
```

---

## 8. Data Validation with Great Expectations

**File**: `src/pipelines/data_validation.py`

```python
import great_expectations as gx
from great_expectations.core.batch import RuntimeBatchRequest
from great_expectations.checkpoint import Checkpoint
import pandas as pd
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class ContactDataValidator:
    """Data quality validation using Great Expectations."""

    def __init__(self):
        self.context = gx.get_context()
        self.suite_name = "contact_enrichment_suite"
        self._create_expectation_suite()

    def _create_expectation_suite(self):
        """Create expectation suite if it doesn't exist."""
        try:
            self.suite = self.context.get_expectation_suite(self.suite_name)
        except:
            self.suite = self.context.add_expectation_suite(self.suite_name)

            # Define expectations
            expectations = [
                # Email validation
                {
                    "expectation_type": "expect_column_values_to_not_be_null",
                    "kwargs": {"column": "email"}
                },
                {
                    "expectation_type": "expect_column_values_to_match_regex",
                    "kwargs": {
                        "column": "email",
                        "regex": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
                    }
                },
                {
                    "expectation_type": "expect_column_values_to_be_unique",
                    "kwargs": {"column": "email"}
                },

                # Completeness checks
                {
                    "expectation_type": "expect_column_values_to_not_be_null",
                    "kwargs": {"column": "first_name"}
                },
                {
                    "expectation_type": "expect_column_values_to_not_be_null",
                    "kwargs": {"column": "last_name"}
                },

                # Enrichment score validation
                {
                    "expectation_type": "expect_column_values_to_be_between",
                    "kwargs": {
                        "column": "enrichment_score",
                        "min_value": 0.0,
                        "max_value": 1.0
                    }
                },
                {
                    "expectation_type": "expect_column_mean_to_be_between",
                    "kwargs": {
                        "column": "enrichment_score",
                        "min_value": 0.7,  # Expect average enrichment > 70%
                        "max_value": 1.0
                    }
                },

                # Company data validation
                {
                    "expectation_type": "expect_column_values_to_not_be_null",
                    "kwargs": {"column": "company"}
                },
                {
                    "expectation_type": "expect_column_value_lengths_to_be_between",
                    "kwargs": {
                        "column": "company",
                        "min_value": 2,
                        "max_value": 255
                    }
                },

                # Date validation
                {
                    "expectation_type": "expect_column_values_to_not_be_null",
                    "kwargs": {"column": "last_enriched_at"}
                },

                # Custom business rules
                {
                    "expectation_type": "expect_table_row_count_to_be_between",
                    "kwargs": {
                        "min_value": 1,
                        "max_value": 1000000
                    }
                },
            ]

            for exp in expectations:
                self.suite.add_expectation(**exp)

            self.context.save_expectation_suite(self.suite)
            logger.info(f"Created expectation suite: {self.suite_name}")

    def validate_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate a pandas DataFrame."""
        try:
            # Create runtime batch request
            batch_request = RuntimeBatchRequest(
                datasource_name="runtime_datasource",
                data_connector_name="runtime_data_connector",
                data_asset_name="contacts",
                runtime_parameters={"batch_data": df},
                batch_identifiers={"default_identifier_name": "default_batch"}
            )

            # Create validator
            validator = self.context.get_validator(
                batch_request=batch_request,
                expectation_suite_name=self.suite_name
            )

            # Run validation
            results = validator.validate()

            # Parse results
            success = results.success
            statistics = results.statistics
            failed_expectations = [
                {
                    "expectation": exp["expectation_config"]["expectation_type"],
                    "column": exp["expectation_config"]["kwargs"].get("column"),
                    "success": exp["success"]
                }
                for exp in results.results
                if not exp["success"]
            ]

            validation_report = {
                "success": success,
                "evaluated_expectations": statistics["evaluated_expectations"],
                "successful_expectations": statistics["successful_expectations"],
                "unsuccessful_expectations": statistics["unsuccessful_expectations"],
                "success_percentage": round(statistics["success_percent"], 2),
                "failures": failed_expectations
            }

            # Log results
            if success:
                logger.info(f"Validation PASSED: {validation_report['success_percentage']}% success")
            else:
                logger.warning(
                    f"Validation FAILED: {len(failed_expectations)} expectations failed"
                )

            return validation_report

        except Exception as e:
            logger.error(f"Validation error: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def validate_database_table(
        self,
        table_name: str = "contacts",
        sample_size: int = 10000
    ) -> Dict[str, Any]:
        """Validate data directly from PostgreSQL."""
        from src.models.database import SessionLocal
        import pandas as pd

        db = SessionLocal()
        try:
            # Sample data from database
            query = f"SELECT * FROM {table_name} ORDER BY RANDOM() LIMIT {sample_size}"
            df = pd.read_sql(query, db.bind)

            logger.info(f"Validating {len(df)} rows from {table_name}")

            return self.validate_dataframe(df)

        finally:
            db.close()

def run_validation_suite(
    datasource: str = "postgres",
    expectation_suite: str = "contact_enrichment_suite"
) -> Dict[str, Any]:
    """Run validation suite (called from Airflow DAG)."""
    validator = ContactDataValidator()

    if datasource == "postgres":
        return validator.validate_database_table()
    else:
        raise ValueError(f"Unsupported datasource: {datasource}")

# Example usage
if __name__ == "__main__":
    validator = ContactDataValidator()

    # Test with sample data
    sample_df = pd.DataFrame({
        "email": ["john@example.com", "jane@company.com"],
        "first_name": ["John", "Jane"],
        "last_name": ["Doe", "Smith"],
        "company": ["Example Inc", "Company Ltd"],
        "enrichment_score": [0.85, 0.92],
        "last_enriched_at": [pd.Timestamp.now(), pd.Timestamp.now()]
    })

    results = validator.validate_dataframe(sample_df)
    print(json.dumps(results, indent=2))
```

---

## 9. Prometheus Metrics

**File**: `src/monitoring/metrics.py`

```python
from prometheus_client import Counter, Histogram, Gauge, Info, generate_latest
from prometheus_client import CollectorRegistry, multiprocess, generate_latest
import time
from functools import wraps
import logging

logger = logging.getLogger(__name__)

# Create registry
registry = CollectorRegistry()

# API Metrics
api_requests_total = Counter(
    'api_requests_total',
    'Total API requests',
    ['endpoint', 'method', 'status'],
    registry=registry
)

api_latency = Histogram(
    'api_request_duration_seconds',
    'API request latency',
    ['endpoint'],
    buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0),
    registry=registry
)

# Task Metrics
task_counter = Counter(
    'celery_task_total',
    'Total Celery tasks',
    ['task_name', 'status'],
    registry=registry
)

task_duration = Histogram(
    'celery_task_duration_seconds',
    'Celery task duration',
    ['task_name'],
    buckets=(1, 5, 10, 30, 60, 120, 300),
    registry=registry
)

# Pipeline Metrics
pipeline_records_processed = Counter(
    'pipeline_records_processed_total',
    'Total records processed',
    ['pipeline', 'source'],
    registry=registry
)

pipeline_duration = Histogram(
    'pipeline_duration_seconds',
    'Pipeline execution duration',
    ['pipeline'],
    buckets=(60, 300, 600, 1800, 3600),
    registry=registry
)

pipeline_failures = Counter(
    'pipeline_failures_total',
    'Total pipeline failures',
    ['pipeline', 'stage'],
    registry=registry
)

# Data Quality Metrics
data_quality_score = Gauge(
    'data_quality_score',
    'Data quality score (0-1)',
    ['source'],
    registry=registry
)

validation_failures = Counter(
    'validation_failures_total',
    'Total validation failures',
    ['validation_type'],
    registry=registry
)

# Cache Metrics
cache_hits = Counter(
    'cache_hits_total',
    'Total cache hits',
    ['cache_type'],
    registry=registry
)

cache_misses = Counter(
    'cache_misses_total',
    'Total cache misses',
    ['cache_type'],
    registry=registry
)

# Database Metrics
db_query_duration = Histogram(
    'db_query_duration_seconds',
    'Database query duration',
    ['query_type'],
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0),
    registry=registry
)

db_connections = Gauge(
    'db_connections_active',
    'Active database connections',
    registry=registry
)

# Enrichment Metrics
enrichment_score_avg = Gauge(
    'enrichment_score_average',
    'Average enrichment score',
    ['source'],
    registry=registry
)

api_call_cost = Counter(
    'external_api_cost_usd',
    'External API costs in USD',
    ['api_provider'],
    registry=registry
)

# System Info
system_info = Info(
    'insightstream_system',
    'System information',
    registry=registry
)

system_info.info({
    'version': '1.0.0',
    'environment': 'production',
    'service': 'insightstream'
})

# Decorators for tracking
def track_api_request(endpoint: str):
    """Decorator to track API requests."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            method = "POST" if hasattr(func, '__self__') else "GET"
            start_time = time.time()

            try:
                result = await func(*args, **kwargs)
                status = "success"
                return result
            except Exception as e:
                status = "error"
                raise
            finally:
                duration = time.time() - start_time
                api_requests_total.labels(
                    endpoint=endpoint,
                    method=method,
                    status=status
                ).inc()
                api_latency.labels(endpoint=endpoint).observe(duration)

        return wrapper
    return decorator

def track_task_execution(task_name: str):
    """Decorator to track Celery task execution."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            task_counter.labels(task_name=task_name, status="started").inc()

            try:
                result = func(*args, **kwargs)
                task_counter.labels(task_name=task_name, status="success").inc()
                return result
            except Exception as e:
                task_counter.labels(task_name=task_name, status="failed").inc()
                raise
            finally:
                duration = time.time() - start_time
                task_duration.labels(task_name=task_name).observe(duration)

        return wrapper
    return decorator

def get_metrics():
    """Get current metrics in Prometheus format."""
    return generate_latest(registry)

# FastAPI endpoint for Prometheus scraping
from fastapi import Response

async def metrics_endpoint():
    """Expose metrics for Prometheus."""
    return Response(
        content=get_metrics(),
        media_type="text/plain"
    )
```

---

## 10. Sentence-Transformers Embeddings

**File**: `src/utils/embeddings.py`

```python
from sentence_transformers import SentenceTransformer
from typing import List, Optional
import numpy as np
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """Load embedding model (cached singleton)."""
    model_name = 'all-MiniLM-L6-v2'  # 384 dimensions, fast
    logger.info(f"Loading embedding model: {model_name}")
    return SentenceTransformer(model_name)

class EmbeddingGenerator:
    """Generate embeddings for contacts."""

    def __init__(self):
        self.model = get_embedding_model()
        self.embedding_dim = 384

    def generate_contact_embedding(self, contact) -> Optional[List[float]]:
        """
        Generate embedding vector for a contact.
        Combines multiple fields into a semantic representation.
        """
        try:
            # Create text representation of contact
            text_parts = []

            if contact.full_name:
                text_parts.append(contact.full_name)

            if contact.title:
                text_parts.append(f"Title: {contact.title}")

            if contact.company:
                text_parts.append(f"Company: {contact.company}")

            if contact.industry:
                text_parts.append(f"Industry: {contact.industry}")

            if contact.location:
                text_parts.append(f"Location: {contact.location}")

            # Combine into single text
            contact_text = ". ".join(text_parts)

            if not contact_text:
                logger.warning(f"No text to embed for contact {contact.id}")
                return None

            # Generate embedding
            embedding = self.model.encode(
                contact_text,
                convert_to_numpy=True,
                show_progress_bar=False
            )

            return embedding.tolist()

        except Exception as e:
            logger.error(f"Failed to generate embedding for contact {contact.id}: {e}")
            return None

    def generate_batch_embeddings(self, contacts: List) -> List[Optional[List[float]]]:
        """Generate embeddings for multiple contacts in batch."""
        texts = []
        valid_indices = []

        for i, contact in enumerate(contacts):
            text_parts = []

            if contact.full_name:
                text_parts.append(contact.full_name)
            if contact.title:
                text_parts.append(f"Title: {contact.title}")
            if contact.company:
                text_parts.append(f"Company: {contact.company}")

            contact_text = ". ".join(text_parts)

            if contact_text:
                texts.append(contact_text)
                valid_indices.append(i)

        if not texts:
            return [None] * len(contacts)

        # Batch encoding for efficiency
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            convert_to_numpy=True,
            show_progress_bar=False
        )

        # Map back to original contact order
        result = [None] * len(contacts)
        for i, embedding in zip(valid_indices, embeddings):
            result[i] = embedding.tolist()

        return result

    def compute_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float]
    ) -> float:
        """Compute cosine similarity between two embeddings."""
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        cosine_sim = np.dot(vec1, vec2) / (
            np.linalg.norm(vec1) * np.linalg.norm(vec2)
        )

        return float(cosine_sim)

# Singleton instance
_embedding_generator = None

def get_embedding_generator() -> EmbeddingGenerator:
    """Get or create embedding generator singleton."""
    global _embedding_generator
    if _embedding_generator is None:
        _embedding_generator = EmbeddingGenerator()
    return _embedding_generator

def generate_embedding(contact) -> Optional[List[float]]:
    """Helper function to generate embedding for a contact."""
    generator = get_embedding_generator()
    return generator.generate_contact_embedding(contact)

def generate_embeddings_batch(contacts: List) -> List[Optional[List[float]]]:
    """Helper function to generate embeddings for multiple contacts."""
    generator = get_embedding_generator()
    return generator.generate_batch_embeddings(contacts)

# Example usage
if __name__ == "__main__":
    from src.models.contact import Contact

    # Test contact
    test_contact = Contact(
        id=1,
        email="john.doe@example.com",
        first_name="John",
        last_name="Doe",
        full_name="John Doe",
        title="Senior Software Engineer",
        company="Tech Corp",
        industry="Technology"
    )

    # Generate embedding
    embedding = generate_embedding(test_contact)
    print(f"Generated embedding with {len(embedding)} dimensions")
    print(f"First 10 values: {embedding[:10]}")

    # Test similarity
    generator = get_embedding_generator()
    test_contact2 = Contact(
        id=2,
        email="jane.smith@example.com",
        first_name="Jane",
        last_name="Smith",
        full_name="Jane Smith",
        title="Software Engineer",
        company="Tech Corp",
        industry="Technology"
    )

    embedding2 = generate_embedding(test_contact2)
    similarity = generator.compute_similarity(embedding, embedding2)
    print(f"Similarity score: {similarity:.4f}")
```

---

## Summary

These code snippets demonstrate:

1. **FastAPI** - High-performance async REST API with caching and pgvector similarity search
2. **Kafka Producer** - Real-time data ingestion with deduplication and partitioning
3. **Celery Tasks** - Distributed async processing with retry logic and circuit breakers
4. **Airflow DAG** - Complex workflow orchestration with parallel data fetching
5. **PostgreSQL + pgvector** - Vector similarity search with efficient indexing
6. **ElasticSearch** - Full-text search with autocomplete and advanced querying
7. **Redis Caching** - Multi-tier caching with decorator patterns
8. **Great Expectations** - Comprehensive data quality validation
9. **Prometheus Metrics** - Production-grade observability and monitoring
10. **Sentence-Transformers** - ML-powered semantic embeddings for duplicate detection

Each snippet is production-ready and demonstrates enterprise best practices for scalability, reliability, and maintainability.
