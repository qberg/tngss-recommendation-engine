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

Detailed Implementation Plan: User-User Matching System
Overview
Build a real-time user matching system that calculates bidirectional compatibility scores for all users when they complete their profile, stores top matches in MongoDB, caches full results in Redis, and handles symmetric updates asynchronously.

Architecture Decisions (Confirmed)
Data Storage

Separate collection: user_matching_scores (isolated from event recommendations)
MongoDB: Top 1000 matches per user (~50M records)
Redis: All 50K scores cached for 24 hours
Symmetric storage: Store both A→B and B→A as separate records

Scoring Algorithm

Asymmetric intent matching: similarity(A.intent_vector, B.org_vector)
Bidirectional score: average(A→B, B→A)
Weights: Start with [0.25, 0.25, 0.5] (same as events)
Filtering: Must have ≥1 overlapping sector AND complementary user types

Real-Time Requirements

POST endpoint returns immediately (<50ms) with task_id
Background Celery task does all calculations (~1.5 seconds)
User polls task status while seeing "Finding matches..." UI
Results available via GET endpoint once complete

Implementation Phases

PHASE 1: Database Foundation
Step 1.1: Create Collection Schema
File: src/database.py
What: Add indexes for new user_matching_scores collection
python# Add to initialize_indexes() function:

- Index: (user_id, score DESC) - sorted retrieval
- Index: (user_id, matched_user_id) UNIQUE - prevent duplicates
- Index: (matched_user_id, updated_at) - reverse lookup for symmetric updates
  Why: These indexes enable:

Fast sorted retrieval of matches
Efficient symmetric update queries
Duplicate prevention

Validation: Run app, check logs for index creation confirmation

Step 1.2: Add Filtering Indexes
File: src/database.py
What: Add indexes on context and org collections for filtering
python# Context builder collection:

- Index: (sector) - for sector overlap filtering
- Index: (looking_to_connect) - for complementarity checks

# Organisation profile collection:

- Index: (profile_type) - for complementarity checks
  Why: Pre-filtering 50K→5K users requires fast lookups on these fields
  Validation: Run app, verify indexes exist with db.collection.getIndexes()

PHASE 2: Filtering Service
Step 2.1: Create UserFilterService
File: src/recommendations/user_filter_service.py (NEW)
What: Service to filter compatible candidates before embeddings
Methods to implement (one at a time):
Method 1: async def get_all_active_user_ids() -> List[str]

Query MongoDB for all users where is_deleted=False
Return list of user IDs
Test: Print count, should be ~50K

Method 2: async def get_user_filter_criteria(user_id: str) -> Dict

Fetch user's sector interests, profile_type, looking_to_connect
Return as dict: {sectors: [...], profile_type: "startup", wants: [...]}
Test: Print criteria for sample user

Method 3: async def filter_compatible_candidates(user_id: str, all_user_ids: List[str]) -> List[str]

Get user's criteria
MongoDB aggregate query:

Match: user_id in all_user_ids
Filter: overlapping sectors (array intersection)
Filter: complementary types (either direction)

Return filtered list
Test: Should reduce 50K → 5-10K

Validation: Run filter for test user, log count before/after

PHASE 3: Matching Service
Step 3.1: Create UserMatchingService
File: src/recommendations/user_matching_service.py (NEW)
What: Core matching logic with vectorized calculations
Dependencies: Reuse existing UserEmbeddingService, EmbeddingService
Methods to implement (one at a time):
Method 1: def calculate_asymmetric_similarity(user_a_embeddings: Dict, user_b_embeddings: Dict) -> Dict

Calculate A→B: similarity(A.intent, B.org)
Calculate B→A: similarity(B.intent, A.org)
Also calculate personal and org similarities for breakdown
Return dict with all scores
Test: Compare two sample users, print scores

Method 2: def calculate_bidirectional_score(a_to_b: float, b_to_a: float) -> float

Return average(a_to_b, b_to_a)
Test: Verify math with sample values

Method 3: async def calculate_matches_vectorized(user_id: str, candidate_ids: List[str]) -> List[Dict]

Load user embeddings (cached or generate)
Batch load candidate embeddings from pickle files
Stack into numpy matrices
Vectorized calculation (like your event matching)
Build list of score dicts
Sort by bidirectional_score descending
Test: Time this - should be ~50ms for 5K candidates

Validation: Compare vectorized vs loop calculation for 10 users, verify same results

Step 3.2: Batch Embedding Loader
File: src/recommendations/user_matching_service.py
What: Efficiently load multiple user embeddings at once
Method: def load_user_embeddings_batch(user_ids: List[str]) -> Dict[str, Dict]

For each user_id, load from pickle file
Return dict: {user_id: {personal: array, org: array, intent: array}}
Skip missing embeddings (log warning)
Test: Load 1000 embeddings, time it (~200ms expected)

Validation: Verify loaded embeddings match individual loads

PHASE 4: Score Storage Service
Step 4.1: Create MatchScoreService
File: src/recommendations/match_score_service.py (NEW)
What: Handle MongoDB operations for user matching scores
Why separate from ScoreService: Different collection, different operations
Methods to implement:
Method 1: async def store_match_scores(user_id: str, scores: List[Dict]) -> bool

Convert to MongoDB documents
Bulk upsert with UpdateOne operations
Handle both directions if symmetric
Test: Store 100 scores, verify in MongoDB

Method 2: async def get_user_matches(user_id: str, limit: int, offset: int) -> List[Dict]

Query with pagination
Sort by score descending
Return serialized results
Test: Fetch matches for test user

Method 3: async def delete_user_matches(user_id: str) -> int

Delete all records where user_id matches
Return count deleted
Test: Create then delete, verify count

Validation: Insert, query, delete cycle with test data

PHASE 5: Redis Caching
Step 5.1: Setup Redis Client
File: src/config.py (modify)
What: Add Redis connection settings
pythonREDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/2")
REDIS_MATCH_CACHE_TTL = 86400 # 24 hours
File: src/cache/redis_client.py (NEW)
What: Redis connection wrapper
Methods:

async def set_user_matches(user_id: str, scores: List[Dict])
async def get_user_matches(user_id: str) -> Optional[List[Dict]]
async def delete_user_matches(user_id: str)

Validation: Set and retrieve test data

PHASE 6: Background Tasks
Step 6.1: Main Calculation Task
File: src/recommendations/tasks.py (modify)
What: Add new Celery task for user matching
Task: @celery_app.task calculate_user_matches(user_id: str)
Logic flow:

Update progress: 10% - "Loading user data..."
Generate user embeddings (or load cached)
Update progress: 30% - "Finding compatible users..."
Get filtered candidates (5-10K)
Update progress: 50% - "Calculating matches..."
Vectorized calculation for all candidates
Update progress: 70% - "Storing results..."
Store top 1000 in MongoDB
Cache all scores in Redis
Update progress: 90% - "Finalizing..."
Queue symmetric updates task
Return success

Validation: Run task manually with test user_id, check logs and database

Step 6.2: Symmetric Update Task
File: src/recommendations/tasks.py (modify)
What: Update reverse matches in background
Task: @celery_app.task update_reverse_matches(user_id: str, match_ids: List[str])
Logic:

For each matched_user_id in top 1000:

Load their embeddings
Recalculate their score with user_id
Update their record where matched_user_id = user_id
Batch operations for efficiency

Why separate task: Don't block main task completion, can retry independently
Validation: Create user A matches, verify user B records update

PHASE 7: API Endpoints
Step 7.1: Calculate Endpoint
File: src/recommendations/router.py (modify)
What: Queue calculation and return immediately
Endpoint: POST /recommendations/user/{user_id}/matches/calculate
Logic:

Validate user exists (10ms)
Check if already calculated (20ms)
If exists: return cache_hit=true
If not: queue Celery task
Return task_id immediately

Response:
json{
"success": true,
"message": "Calculation queued",
"task_id": "abc-123",
"cache_hit": false
}
Validation: Call endpoint, verify task queues

Step 7.2: Retrieval Endpoint
File: src/recommendations/router.py (modify)
What: Get user's matches with pagination
Endpoint: GET /recommendations/user/{user_id}/matches?limit=20&offset=0
Logic:

If offset < 1000: Query MongoDB
If offset >= 1000: Check Redis cache
If not found: Return 404 with "Calculate first" message

Response:
json[
{
"user_id": "...",
"matched_user_id": "...",
"score": 85,
"similarity_breakdown": {...}
}
]
Validation: Calculate matches, then retrieve them

Step 7.3: Task Status Endpoint
File: src/recommendations/router.py (already exists)
What: Reuse existing task status endpoint
Endpoint: GET /recommendations/task/{task_id}/status
No changes needed - already handles progress updates
Validation: Poll while task runs, see progress updates

PHASE 8: Testing & Validation
Step 8.1: Unit Tests
For each service:

Test filtering logic with mock data
Test vectorized calculations accuracy
Test score storage and retrieval

Step 8.2: Integration Tests
End-to-end flow:

Create test user with complete profile
POST /calculate, get task_id
Poll /task/status until complete
GET /matches, verify results
Check MongoDB for top 1000
Check Redis for full cache

Step 8.3: Performance Tests
Measure:

User embedding generation: <200ms
Filtering 50K→5K: <50ms
Vectorized calculation 5K: <50ms
MongoDB bulk insert 1000: <250ms
Total task time: <1.5 seconds

File Structure Summary
src/
├── recommendations/
│ ├── user_filter_service.py # NEW - Phase 2
│ ├── user_matching_service.py # NEW - Phase 3
│ ├── match_score_service.py # NEW - Phase 4
│ ├── tasks.py # MODIFY - Phase 6
│ └── router.py # MODIFY - Phase 7
├── cache/
│ └── redis_client.py # NEW - Phase 5
├── database.py # MODIFY - Phase 1
└── config.py # MODIFY - Phase 5
