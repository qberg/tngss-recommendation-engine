"""Rest API client wrapper for fetching events"""

import asyncio
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

import httpx

from src.config import settings
from src.utils.setup_logger import setup_logger

logger = setup_logger(__name__, "logs/events_client.log")


class EventsClient:
    """Wrapper for Payload events REST API"""

    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url or settings.PAYLOAD_CMS_URL
        self.client = httpx.AsyncClient()

    async def close(self):
        """Close the HTTP Client"""
        await self.client.aclose()

    async def get_event(
        self, event_id: str, depth: int = 2
    ) -> Optional[Dict[str, Any]]:
        """Get single event by ID"""
        try:
            params = {"depth": str(depth)}
            query_string = urlencode(params)

            url = f"{self.base_url}/api/events/{event_id}?{query_string}"
            logger.info(f"[...] Fetching event from: {url}")

            response = await self.client.get(url)
            response.raise_for_status()

            event_data = response.json()
            logger.info(f"[SUCCESS] Event with id:{event_id} fetched sucessfully")
            return event_data

        except Exception as e:
            logger.error(f"[FAILED] Fetching event with id {event_id} failed: {e}")
            return None

    async def get_all_public_events(self, batch_size: int) -> List[Dict[str, Any]]:
        """Get all public events by paginating through all pages"""
        batch_size = batch_size or 100

        start_time = time.time()
        all_events = []
        page = 1

        logger.info(f"[START] Fetching all public events with batch_size={batch_size}")
        while True:
            try:
                params = {
                    "where[isPublic][equals]": "true",
                    "limit": batch_size,
                    "page": page,
                    "depth": 2,
                }
                query_string = urlencode(params)
                url = f"{self.base_url}/api/events?{query_string}"
                # url = f"{self.base_url}/api/events?where%5BisPublic%5D%5Bequals%5D=true&limit={batch_size}&page={page}&depth=2"
                logger.info(
                    f"[...] Fetching page {page} with {batch_size} events per page"
                )

                response = await self.client.get(url)
                response.raise_for_status()
                data = response.json()

                docs = data.get("docs", [])

                if not docs:
                    break

                all_events.extend(docs)
                logger.info(
                    f"[SUCCESS] Page {page}: Got {len(docs)} events. Total so far: {len(all_events)}"
                )

                if not data.get("hasNextPage", False):
                    break

                page += 1

            except Exception as e:
                logger.error(f"[FAILED] Error fetching page {page}: {e}")
                break

        end_time = time.time()
        exeution_time = end_time - start_time

        logger.info(
            f"[COMPLETE] Total events: {len(all_events)} | Time taken: {exeution_time:.2f} seconds"
        )
        logger.info(f"[STATS] Speed: {len(all_events)/exeution_time:.1f} events/second")

        return all_events


async def main():
    """Test the Events Client"""
    client = EventsClient()
    try:
        logger.info("[***] Starting the EventsClient test")

        print("#" * 50)
        print("# Test 1")
        print("#" * 50)
        event_id = "68d23b95502afe93dad81501"

        result = await client.get_event(event_id)

        if result:
            logger.info(
                f"[SUCCESS] Fetched event with title: {result.get('title', 'No Title')}"
            )

        print("#" * 50)
        print("# Test 2")
        print("#" * 50)
        for batch_size in [50, 100]:
            logger.info(f"[TEST] Testing with batch_size={batch_size}")
            all_events = await client.get_all_public_events(batch_size=batch_size)
            print("-" * 50)
            print(f"Batch size {batch_size}: {len(all_events)} events")
            print("-" * 50)

    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
