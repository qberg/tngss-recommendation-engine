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
```
