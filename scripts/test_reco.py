"""
Script to test and benchmark the user recommendation query performance.
Run this to measure timings before and after optimizations.
"""

import asyncio
import time
from typing import Dict, List

from pymongo import ASCENDING, DESCENDING

from src.config import settings
from src.database import close_mongo_connection, connect_to_mongo, get_database
from src.utils.setup_logger import setup_logger

logger = setup_logger(__name__, "logs/performance_test.log")


class RecommendationPerformanceTester:
    def __init__(self):
        self.db = None
        self.results = []

    async def setup(self):
        """Initialize database connection"""

        # Connect to MongoDB first
        await connect_to_mongo()
        self.db = get_database()
        logger.info("[SETUP] Database connection established")

    async def get_sample_user_id(self):
        """Get a user ID that has AI scores"""
        user_score = await self.db[settings.USER_RECOMMENDATIONS_COLLECTION].find_one(
            {"score": {"$exists": True, "$gt": 0}}, {"user_id": 1}
        )
        if user_score:
            return user_score["user_id"]

        # Fallback: get any user
        user = await self.db["login_info"].find_one(
            {"role": "user", "is_deleted": False}, {"_id": 1}
        )
        return user["_id"] if user else None

    async def test_original_query(self, user_id, limit=50):
        """Test the original lookup-based approach"""
        logger.info("[TEST] Running ORIGINAL query (lookup on each user)...")

        search_obj = {
            "is_deleted": False,
            "is_email_verified": True,
            "role": "user",
            "_id": {"$ne": user_id},
        }

        pipeline = [
            {"$match": search_obj},
            # Simulate other lookups (simplified)
            {
                "$lookup": {
                    "from": settings.USER_RECOMMENDATIONS_COLLECTION,
                    "let": {"targetId": "$_id", "userId": user_id},
                    "pipeline": [
                        {
                            "$match": {
                                "$expr": {
                                    "$and": [
                                        {"$eq": ["$user_id", "$$userId"]},
                                        {"$eq": ["$matched_user_id", "$$targetId"]},
                                    ]
                                }
                            }
                        },
                        {"$project": {"score": 1}},
                        {"$limit": 1},
                    ],
                    "as": "ai_score_details",
                }
            },
            {
                "$addFields": {
                    "ai_score": {
                        "$ifNull": [{"$arrayElemAt": ["$ai_score_details.score", 0]}, 0]
                    }
                }
            },
            {"$sort": {"ai_score": -1}},
            {"$limit": limit},
            {"$project": {"user_id": "$_id", "first_name": 1, "ai_score": 1}},
        ]

        start_time = time.time()
        cursor = await self.db["login_info"].aggregate(pipeline, allowDiskUse=True)
        results = await cursor.to_list(length=None)
        end_time = time.time()

        execution_time = (end_time - start_time) * 1000  # Convert to ms

        logger.info(
            f"[RESULT] Original query: {execution_time:.2f}ms, {len(results)} results"
        )
        return {
            "method": "original_lookup",
            "time_ms": execution_time,
            "result_count": len(results),
            "results": results[:5],  # Sample results
        }

    async def test_optimized_query(self, user_id, limit=50):
        """Test the optimized approach (query ai_score collection first)"""
        logger.info("[TEST] Running OPTIMIZED query (start from ai_score)...")

        pipeline = [
            {"$match": {"user_id": user_id, "score": {"$exists": True, "$gt": 0}}},
            {"$sort": {"score": -1}},
            {"$limit": limit},
            {
                "$lookup": {
                    "from": "login_info",
                    "let": {"matchedUserId": "$matched_user_id"},
                    "pipeline": [
                        {
                            "$match": {
                                "$expr": {"$eq": ["$_id", "$$matchedUserId"]},
                                "is_deleted": False,
                                "is_email_verified": True,
                                "role": "user",
                            }
                        },
                        {"$project": {"first_name": 1, "profile_image": 1}},
                    ],
                    "as": "user_details",
                }
            },
            {"$unwind": {"path": "$user_details", "preserveNullAndEmptyArrays": False}},
            {
                "$project": {
                    "user_id": "$matched_user_id",
                    "first_name": "$user_details.first_name",
                    "ai_score": "$score",
                }
            },
        ]

        start_time = time.time()
        cursor = await self.db[settings.USER_RECOMMENDATIONS_COLLECTION].aggregate(
            pipeline, allowDiskUse=True
        )
        results = await cursor.to_list(length=None)
        end_time = time.time()

        execution_time = (end_time - start_time) * 1000

        logger.info(
            f"[RESULT] Optimized query: {execution_time:.2f}ms, {len(results)} results"
        )
        return {
            "method": "optimized_reverse",
            "time_ms": execution_time,
            "result_count": len(results),
            "results": results[:5],
        }

    async def test_with_explain(self, user_id):
        """Run explain plan to see index usage"""
        logger.info("[EXPLAIN] Analyzing query execution plan...")

        pipeline = [
            {"$match": {"user_id": user_id, "score": {"$exists": True}}},
            {"$sort": {"score": -1}},
            {"$limit": 50},
        ]

        # Use command instead of aggregate().explain()
        explain = await self.db.command(
            "aggregate",
            settings.USER_RECOMMENDATIONS_COLLECTION,
            pipeline=pipeline,
            explain=True,
        )

        # Extract key metrics from explain output
        if "stages" in explain:
            stages = explain["stages"]
            for stage in stages:
                if "$cursor" in stage:
                    query_planner = stage["$cursor"].get("queryPlanner", {})
                    winning_plan = query_planner.get("winningPlan", {})
                    index_name = winning_plan.get("indexName", "COLLECTION SCAN")
                    logger.info(f"[INDEX USED] {index_name}")

                    exec_stats = stage["$cursor"].get("executionStats", {})
                    if exec_stats:
                        logger.info(
                            f"[DOCS EXAMINED] {exec_stats.get('totalDocsExamined', 0)}"
                        )
                        logger.info(f"[DOCS RETURNED] {exec_stats.get('nReturned', 0)}")
                        logger.info(
                            f"[EXECUTION TIME] {exec_stats.get('executionTimeMillis', 0)}ms"
                        )
        elif "explainVersion" in explain:
            # New explain format (MongoDB 4.4+)
            query_planner = explain.get("queryPlanner", {})
            winning_plan = query_planner.get("winningPlan", {})
            logger.info(f"[PLAN] {winning_plan}")

        return explain

    async def check_indexes(self):
        """Verify indexes exist"""
        logger.info("[INDEXES] Checking index status...")

        collections = {
            "login_info": "login_info",
            "user_ai_score": settings.USER_RECOMMENDATIONS_COLLECTION,
        }

        for name, collection in collections.items():
            indexes = await self.db[collection].index_information()
            logger.info(f"[{name}] Indexes: {list(indexes.keys())}")

            # Check for compound index on user_ai_score
            if name == "user_ai_score":
                has_compound = any(
                    "user_id" in str(idx) and "matched_user_id" in str(idx)
                    for idx in indexes.values()
                )
                logger.info(
                    f"[{name}] Has user_id + matched_user_id index: {has_compound}"
                )

    async def run_stress_test(self, user_id, iterations=10):
        """Run multiple iterations to get average timings"""
        logger.info(f"[STRESS TEST] Running {iterations} iterations...")

        original_times = []
        optimized_times = []

        for i in range(iterations):
            logger.info(f"[ITERATION {i+1}/{iterations}]")

            # Test original
            result = await self.test_original_query(user_id, limit=50)
            original_times.append(result["time_ms"])

            # Small delay
            await asyncio.sleep(0.1)

            # Test optimized
            result = await self.test_optimized_query(user_id, limit=50)
            optimized_times.append(result["time_ms"])

            await asyncio.sleep(0.1)

        # Calculate statistics
        avg_original = sum(original_times) / len(original_times)
        avg_optimized = sum(optimized_times) / len(optimized_times)
        improvement = ((avg_original - avg_optimized) / avg_original) * 100

        logger.info("\n" + "=" * 60)
        logger.info("[STRESS TEST RESULTS]")
        logger.info(f"Original Average: {avg_original:.2f}ms")
        logger.info(f"Optimized Average: {avg_optimized:.2f}ms")
        logger.info(f"Improvement: {improvement:.1f}% faster")
        logger.info(f"Speed multiplier: {avg_original/avg_optimized:.2f}x")
        logger.info("=" * 60 + "\n")

        return {
            "original_avg": avg_original,
            "optimized_avg": avg_optimized,
            "improvement_percent": improvement,
            "original_times": original_times,
            "optimized_times": optimized_times,
        }

    async def run_all_tests(self):
        """Run complete test suite"""
        logger.info("=" * 60)
        logger.info("STARTING PERFORMANCE TESTS")
        logger.info("=" * 60 + "\n")

        await self.setup()

        # Check indexes first
        await self.check_indexes()

        # Get sample user
        user_id = await self.get_sample_user_id()
        if not user_id:
            logger.error("[ERROR] No valid user found for testing")
            return

        logger.info(f"[TEST USER] {user_id}\n")

        # Run explain plan
        await self.test_with_explain(user_id)

        # Run stress test
        stress_results = await self.run_stress_test(user_id, iterations=10)

        logger.info("\n[COMPLETE] All tests finished")
        return stress_results


async def main():
    """Main entry point"""
    tester = RecommendationPerformanceTester()

    try:
        results = await tester.run_all_tests()

        if results:
            print("\n" + "=" * 60)
            print("FINAL SUMMARY")
            print("=" * 60)
            print(f"Original Query Avg: {results['original_avg']:.2f}ms")
            print(f"Optimized Query Avg: {results['optimized_avg']:.2f}ms")
            print(f"Performance Gain: {results['improvement_percent']:.1f}%")
            print("=" * 60)
    finally:
        # Clean up database connection
        await close_mongo_connection()


if __name__ == "__main__":
    asyncio.run(main())
