```bash
python -c "from src.config import settings; print(settings.MONGODB_URL)"

# Check db connection
python -m tests.db.connect
```
