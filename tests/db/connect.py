import asyncio

from src.database import (connect_to_mongo, get_events_collection,
                          get_users_collection)


async def test_database():
    try:
        # Test connection
        print("Testing database connection...")
        await connect_to_mongo()
        print("[SUCCESS] Connected successfully!")

        # Test collections
        users = get_users_collection()
        events = get_events_collection()

        # Count documents
        user_count = await users.count_documents({})
        event_count = await events.count_documents({})

        print(f"[SUCCESS] Users collection: {user_count} documents")
        print(f"[SUCCESS] Events collection: {event_count} documents")

    except Exception as e:
        print(f"[ERROR] {e}")


if __name__ == "__main__":
    asyncio.run(test_database())
