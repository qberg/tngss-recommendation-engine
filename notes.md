AI-Powered Event Recommendation System - Complete Technical Summary
Project Overview
Built a production-ready recommendation system that matches users with events based on semantic similarity of their profiles. The system handles 50K+ users and generates personalized event recommendations in real-time.

Architecture Components

1. Technology Stack

Backend: FastAPI (Python) with async/await
Database: MongoDB (async client) - 4 collections per user
Cache Layer: Redis 6+ with ACL authentication
Task Queue: Celery 5.5.3 with Redis broker
Embeddings: OpenAI text-embedding-3-small (1536 dimensions)
Vector Storage: Local pickle files with metadata
Deployment: AWS Ubuntu, Gunicorn (4 workers), Supervisor, Nginx

2. Data Model
   User data spans 4 MongoDB collections:

login_info: Authentication, basic profile (name, email, org reference)
user_profile: Personal details (designation, bio, location, gender, sector)
organisation_profile: Company info (name, type, sector, stage, funding, offerings)
context_builder: Intent data (looking_to_connect, looking_to_meet, sector interests)

User types (9 categories):
startup, investors, aspirants_individuals, government, mentor_sme, incubation_acceleration, industry_corporate, ecosystem_service_provider, others

Core Algorithm
Multi-Vector Embedding Strategy
Each user gets 3 separate embeddings generated from different aspects:

Personal Vector: Name, designation, organization, bio, location, gender context
Organizational Vector: Company details, sector, stage, business model, offerings, funding
Intent Vector: What they're seeking, who they want to connect with, sector interests

Event Vector: Single embedding from event title, description, speakers, topics, date
Similarity Calculation (Vectorized)
python# Stack user embeddings: (3, 1536)
user_matrix = [personal_emb, org_emb, intent_emb]

# Stack event embeddings: (N_events, 1536)

event_matrix = [event1_emb, event2_emb, ...]

# Batch compute: (3, N_events)

similarity_matrix = user_matrix @ event_matrix.T

# Weighted average

weights = [0.25, 0.25, 0.5] # personal, org, intent
final_scores = weights @ similarity_matrix

# Normalize to 10-95% range

percentage_scores = normalize(final_scores)
Performance: 224 events processed in ~1.5ms via NumPy vectorization

Caching & Invalidation
3-Tier Caching Strategy
Tier 1: User Embeddings (Pickle Files)

Stored: data/embeddings/users/{user_id}.pkl
Contains: embeddings + metadata (source texts, timestamps, raw data hashes)
Invalidation: Content-hash based - only regenerates if embedding-relevant fields change

Tracks: designation, bio, sector, goals, looking_for fields
Ignores: last_name, phone_number, non-embedding fields

Cache check: 88ms including hash computation

Tier 2: Event Embeddings (Pickle Files)

Stored: data/embeddings/events/{event_id}.pkl
Pre-computed: Batch job every 6 hours (224 events in ~2 minutes)
Invalidation: Stale if event modified after embedding created

Tier 3: Recommendation Scores (MongoDB)

Collection: ai_score with compound indexes
Stored: user_id, reference_id, score, similarity_breakdown, updated_at
TTL: 24 hours
Indexes:

(user_id, reference_type, updated_at) - fetch recent scores
(user_id, reference_type, score) - sorted retrieval
(user_id, reference_id, reference_type) - unique constraint

Smart Cache Invalidation
python# On each request, checks:

1. Do scores exist? (MongoDB query)
2. If yes, get user raw data (4 parallel MongoDB queries: 40ms)
3. Generate current embedding texts (profile service)
4. Hash current texts
5. Compare with cached hash
6. If different → regenerate user embeddings only
7. If same → return cached scores

API Endpoints
GET /recommendations/user/{user_id}/events

Returns: Top N stored recommendations
Response time: <100ms (MongoDB indexed lookup)
Returns 404 if not calculated yet

POST /recommendations/user/{user_id}/events/calculate

Queues Celery task for calculation
Returns: task_id immediately (<50ms)
Logic:

Validates user exists
Checks cache (with content hash validation)
If valid cache → returns cached scores
If stale → queues Celery task with force_user_regenerate=True

Query params:

force_recalculate: Regenerate all (user + events)
force_user_regenerate: Regenerate user embeddings only

GET /recommendations/task/{task_id}/status

Checks Celery task progress
States: pending, processing, completed, failed
Returns: progress percentage, current step, result

Background Processing (Celery)
Task: calculate_user_recommendations
Execution flow:

Create fresh MongoDB connection (Celery worker isolation)
Generate/retrieve user embeddings (with force flags)
Load cached event embeddings (224 events: 48ms)
Vectorized similarity calculation (1.5ms)
Build scores list (0.13ms)
Sort and normalize to percentages
Bulk upsert to MongoDB (258ms for 224 scores)
Close connection

Performance: 430ms total for new user with cached events
Error handling:

Max retries: 3
Retry delay: 10 seconds
Automatic retry on failure
Progress updates via task.update_state()

Celery Configuration

Broker: Redis DB 0
Backend: Redis DB 1
Concurrency: 8 workers (2x CPU cores for I/O-bound tasks)
Serialization: JSON
Time limits: 300s hard, 240s soft
Worker refresh: Every 100 tasks (prevent memory leaks)

Database Optimization
MongoDB Indexes
Profile collections:

user*id_1 on user_profile (ascending, unique)
user_id*-1 on context_builder (descending, unique)

Recommendations collection:
python[
(user_id, reference_type, updated_at DESC),
(user_id, reference_type, score DESC),
(user_id, reference_id, reference_type) UNIQUE
]
Query optimization:

Parallel fetching: 4 collections in 40ms (asyncio.gather)
Connection pooling: 50 max, 10 min
Query projections: Fetch only needed fields (not implemented yet)

Write Performance
Before optimization: 1731ms for 224 scores (129 scores/sec)
After removing duplicate indexes: 258ms (869 scores/sec)
Improvement: 6.7x faster

Text Generation (ProfileService)
Personal Text
Combines: name, designation, organization, sector, bio, location, gender pronouns
Example: "He works as Product Manager at TechCorp. Their organization focuses in fintech_insurtech sector as startup. Background: Building payment solutions for SMBs. He is currently exploring opportunities in fintech_insurtech."
Organizational Text
Combines: company name, type, sector, about, offerings, business model, stage, team size, funding, revenue
Example: "TechCorp focuses on helping small businesses accept payments digitally. The company offers subsidized_rates, mentorship_advisory. It operates using B2B, SaaS business models. The organization is at seed_stage with a team of 12 people having raised 10L-50L so far."
Intent Text
Combines: looking_for, looking_to_connect, looking_to_meet, sector interests, offerings
Example: "They are currently seeking funding, mentorship. They want to connect with investors, mentor_sme. They are particularly interested in meeting Angel Investor, Managing Partner. Their main sector interests include fintech_insurtech, ai_ml_iot."
Token limits: Each text kept under embedding model limits via tiktoken

Deployment Configuration
Supervisor (Process Management)
FastAPI (Gunicorn):
iniworkers: 4
worker*class: uvicorn.workers.UvicornWorker
bind: 127.0.0.1:8000
timeout: 60s
Celery Worker:
iniconcurrency: 8
loglevel: info
autostart: true
autorestart: true
logs: /var/log/celery/
Management:
bashsupervisorctl restart fastapi-workers:*
supervisorctl status fastapi-workers:\_
Redis Configuration

ACL enabled (Redis 6+)
User: celery*user with password
Permissions: ~* &\_ +@all (all keys, channels, commands)
Memory: 256MB with allkeys-lru eviction
Bind: localhost only

System Resources (4-core AWS instance)

CPU: 4 vCPUs (Intel Xeon 3.0GHz)
Allocation: 4 Gunicorn + 8 Celery = 12 total processes
Utilization: ~60-70% under load

Performance Metrics
Latency Breakdown (New User)
User data fetch (4 collections): 40ms
User embedding generation: 100-200ms
Event embeddings load: 48ms
Vectorized calculation: 1.5ms
Score list construction: 0.13ms
Database write: 258ms

---

Total: 430ms
Capacity Analysis
Current (8 Celery workers):

Concurrent tasks: 8
Task duration: 430ms
Throughput: ~1100 new users/minute

With 80% cache hit rate:

Cached responses: <100ms
Effective capacity: 1900+ requests/minute

Database Performance

Read latency: 10-30ms (indexed queries)
Write latency: 258ms (bulk upsert 224 scores)
Connection pool utilization: ~30%

Code Organization (Netflix Dispatch Pattern)
src/
├── recommendations/
│ ├── router.py # FastAPI endpoints
│ ├── service.py # Orchestration layer
│ ├── user_embedding_service.py
│ ├── event_embedding_service.py
│ ├── score_service.py # DB operations
│ ├── batch_service.py # Bulk operations
│ ├── tasks.py # Celery tasks
│ ├── celery_config.py # Celery setup
│ └── schemas.py # Pydantic models
├── profiles/
│ ├── service.py # Text generation
│ └── schemas.py # User data models
├── embeddings/
│ └── service.py # OpenAI API wrapper
├── vector_store/
│ └── service.py # Pickle cache management
├── events/
│ ├── client.py # External events API
│ └── service.py # Event processing
├── database.py # MongoDB connection + indexes
├── config.py # Environment settings
└── main.py # FastAPI app

Key Design Decisions

Multi-vector embeddings: Captures different aspects of user identity for nuanced matching
Pickle cache over vector DB: Simple, fast, no additional infrastructure for 50K scale
Content-hash invalidation: Only regenerates when actual embedding inputs change
Celery for async: Handles concurrent load, retry logic, progress tracking
Vectorized numpy operations: 10-20x faster than loops for similarity calculation
Pre-compute events: Events change slowly, compute once and reuse
Real-time users: Users complete profiles and expect instant recommendations
Separate user/event regeneration: Don't recompute events when user profile changes
MongoDB for scores: Indexes enable fast sorted retrieval, TTL for freshness

Production Hardening
Implemented:

Connection pooling (MongoDB, Redis)
Graceful degradation (cache misses handled)
Retry logic (Celery auto-retry)
Monitoring logs (structured logging)
Process supervision (Supervisor)
Index optimization (removed duplicates)
Vectorized computation (NumPy)
Parallel queries (asyncio.gather)

Future optimizations:

Query projections (fetch only needed fields)
Redis caching for GET endpoints
Prometheus metrics
Event embedding cron job automation
Task deduplication (prevent duplicate calculations)

Critical Gotchas Solved

Index conflicts: Had duplicate indexes slowing writes by 6.7x
Cache invalidation: Initially regenerated on any field change; now only on embedding-relevant fields
Async in Celery: Required sync wrapper around async code
Connection isolation: Celery workers need separate DB connections
Supervisor groups: Needed both services in same group for unified management
Redis ACL: Redis 6+ uses ACL instead of requirepass
Vector direction: Index naming (_1 vs _-1) must match actual direction

This system handles 50K users, generates recommendations in <500ms, and scales to 1900+ req/min with caching.
