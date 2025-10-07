```bash
python -c "from src.config import settings; print(settings.MONGODB_URL)"

# Check db connection
python -m tests.db.connect


 uvicorn src.main:app --reload

http://localhost:8000/recommendations/user/68d2a8d7d84108aa7b141508/events?max_events=5
http://localhost:8000/recommendations/user/68d2a8d7d84108aa7b141508/events/calculate

sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl start fastapi-workers:fastapi
sudo supervisorctl status fastapi


sudo supervisorctl tail -f fastapi-workers:fastapi
ps aux | grep -E "(gunicorn|uvicorn)" | grep -v grep


celery -A src.recommendations.celery_config worker --loglevel=info --concurrency=4


sudo supervisorctl restart fastapi-workers:*  # Restart both
sudo supervisorctl status fastapi-workers:*   # Check status
tail -f /var/log/celery/worker.log           # Monitor Celery
tail -f /var/log/fastapi/error.log           # Monitor API


celery -A src.recommendations.celery_config worker --loglevel=info --concurrency=4
 sudo supervisorctl tail -f fastapi-workers:celery_worker
```

# On each request, checks:

1. Do scores exist? (MongoDB query)
2. If yes, get user raw data (4 parallel MongoDB queries: 40ms)
3. Generate current embedding texts (profile service)
4. Hash current texts
5. Compare with cached hash
6. If different → regenerate user embeddings only
7. If same → return cached scores

## User recommendations collection justification:

1. First Index: user_id + score
   - For queries of type show me top matches of this user
   - score is sorted descending, Mongo can use the index directly
2. Second Index: user_id + mathced_user_id, unique
   - No duplicate (A,B) pairs
   - O(1) lookup for if user A is matched with user B
3. Third Index: matched_user_id + updatedAt
   - Who has this user as a match
   - For symmetric updates
