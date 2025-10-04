"""Service for calculating user-to-user match scores."""

import time
from typing import Dict, List

import numpy as np
from pymongo.asynchronous.database import AsyncDatabase

from src.recommendations.user_embedding_service import UserEmbeddingService
from src.utils.setup_logger import setup_logger

logger = setup_logger(__name__, "logs/user_matching_service.log")


class UserMatchingService:
    """Handles user-to-user matching calculations."""

    def __init__(self, db: AsyncDatabase):
        self.db = db
        self.user_embedding_service = UserEmbeddingService(db)

    def load_user_embeddings_batch(
        self, user_ids: List[str]
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Load embeddings for multiple users from pickle files.
        Returns dict: {user_id: {personal: array, org: array, intent: array}}
        Skips users with missing embeddings.
        """
        method = self.load_user_embeddings_batch.__name__
        try:
            start_time = time.perf_counter()

            embeddings_map = {}
            missing_count = 0

            for user_id in user_ids:
                user_embeddings = (
                    self.user_embedding_service.vector_store.get_user_embeddings(
                        user_id
                    )
                )

                if user_embeddings:
                    embeddings_map[user_id] = user_embeddings
                else:
                    missing_count += 1

            elapsed = (time.perf_counter() - start_time) * 1000

            logger.info(
                f"[{method}] Loaded {len(embeddings_map)} embeddings in {elapsed:.2f}ms "
            )

            return embeddings_map

        except Exception as e:
            logger.error(f"[{method}] [FAILED] Error loading batch embeddings: {e}")
            raise

    def normalize_scores_to_percentage(self, matches: List[Dict]) -> List[Dict]:
        """
        Normalize similarity scores to percentage range (10-95%) using vectorization.
        """
        if not matches:
            return matches

        scores = np.array([m["similarity_score"] for m in matches])
        min_score = scores.min()
        max_score = scores.max()

        if max_score == min_score:
            # All scores identical
            percentage_scores = np.full(len(scores), 52.5)
        else:
            # Vectorized normalization
            percentage_scores = (
                10 + ((scores - min_score) / (max_score - min_score)) * 85
            )

        # Assign back to matches
        for i, match in enumerate(matches):
            match["percentage_score"] = round(percentage_scores[i])

        return matches

    def calculate_assymetric_similarity(
        self,
        user_a_embeddings: Dict[str, np.ndarray],
        user_b_embeddings: Dict[str, np.ndarray],
    ) -> Dict[str, float]:
        """
        Calculate asymmetric similarity between two users.
        A→B: How relevant is B to A's intent
        B→A: How relevant is A to B's intent

        Returns dict with individual scores and bidirectional average.
        """
        # A->B: A's intent vs B's org (what A wants vs what B offers)
        a_to_b = float(np.dot(user_a_embeddings["intent"], user_b_embeddings["org"]))

        # B-> A (what B wants vs what A offers)
        b_to_a = float(np.dot(user_b_embeddings["intent"], user_a_embeddings["org"]))

        # Similarities between them
        personal_sim = float(
            np.dot(user_a_embeddings["personal"], user_b_embeddings["personal"])
        )
        org_sim = float(np.dot(user_a_embeddings["org"], user_b_embeddings["org"]))

        # Bidirectional score
        bidirectional = (a_to_b + b_to_a) / 2.0

        return {
            "a_to_b": a_to_b,
            "b_to_a": b_to_a,
            "personal": personal_sim,
            "org": org_sim,
            "bidirectional": bidirectional,
        }

    async def calculate_matches_vectorized(
        self, user_id: str, candidate_ids: List[str]
    ) -> List[Dict]:
        """
        Calculate match scores for user against all candidates using vectorization.
        Returns list of score dicts sorted by bidirectional score.
        """
        method = self.calculate_matches_vectorized.__name__
        try:
            start_time = time.perf_counter()

            t1 = time.perf_counter()
            user_embeddings = (
                await self.user_embedding_service.get_or_generate_user_embeddings(
                    user_id, force_regenerate=False
                )
            )
            user_emb_time = (time.perf_counter() - t1) * 1000

            t2 = time.perf_counter()
            candidate_embeddings = self.load_user_embeddings_batch(candidate_ids)
            load_time = (time.perf_counter() - t2) * 1000

            if not candidate_embeddings:
                logger.warning(f"[{method}] No candidate embeddings found")
                return []

            valid_candidate_ids = list(candidate_embeddings.keys())

            # Stack candidate embeddings into matrices
            t3 = time.perf_counter()
            personal_matrix = np.vstack(
                [candidate_embeddings[cid]["personal"] for cid in valid_candidate_ids]
            )
            org_matrix = np.vstack(
                [candidate_embeddings[cid]["org"] for cid in valid_candidate_ids]
            )
            intent_matrix = np.vstack(
                [candidate_embeddings[cid]["intent"] for cid in valid_candidate_ids]
            )

            # User vectors
            user_personal = user_embeddings["personal"]
            user_org = user_embeddings["org"]
            user_intent = user_embeddings["intent"]

            # Vectorized calculations
            # A→B: user's intent vs candidates' org
            a_to_b_scores = user_intent @ org_matrix.T

            # B→A: candidates' intent vs user's org
            b_to_a_scores = intent_matrix @ user_org

            # Other similarities
            personal_scores = user_personal @ personal_matrix.T
            org_scores = user_org @ org_matrix.T

            # Bidirectional scores
            bidirectional_scores = (a_to_b_scores + b_to_a_scores) / 2.0

            calc_time = (time.perf_counter() - t3) * 1000

            # Build results
            t4 = time.perf_counter()
            results = []
            for idx, candidate_id in enumerate(valid_candidate_ids):
                results.append(
                    {
                        "user_id": user_id,
                        "matched_user_id": candidate_id,
                        "similarity_score": float(bidirectional_scores[idx]),
                        "similarity_breakdown": {
                            "a_to_b": float(a_to_b_scores[idx]),
                            "b_to_a": float(b_to_a_scores[idx]),
                            "personal": float(personal_scores[idx]),
                            "org": float(org_scores[idx]),
                            "bidirectional": float(bidirectional_scores[idx]),
                        },
                    }
                )

            # Sort by bidirectional score descending
            results.sort(key=lambda x: x["similarity_score"], reverse=True)
            results = self.normalize_scores_to_percentage(results)

            build_time = (time.perf_counter() - t4) * 1000
            elapsed = (time.perf_counter() - start_time) * 1000

            logger.info(
                f"[{method}] Calculated {len(results)} matches in {elapsed:.2f}ms "
                f"(user_emb: {user_emb_time:.2f}ms, load: {load_time:.2f}ms, "
                f"calc: {calc_time:.2f}ms, build: {build_time:.2f}ms)"
            )

            return results
        except Exception as e:
            logger.error(f"[{method}] Error calculating matches: {e}")
            raise e


async def test():

    await connect_to_mongo()
    db = get_database()
    service = UserMatchingService(db)

    filter_service = UserFilterService(db)
    user_ids = await filter_service.get_all_active_user_ids()

    print(f"Total users: {len(user_ids)}")

    test_ids = user_ids[:10]

    print("+" * 60)
    print("Test 1: Embeddings Map for existing")
    print("+" * 60)

    embeddings_map = service.load_user_embeddings_batch(test_ids)

    print(f"\nLoaded embeddings for {len(embeddings_map)}/{len(test_ids)} users")
    if embeddings_map:
        sample_id = list(embeddings_map.keys())[0]
        sample = embeddings_map[sample_id]
        print(f"\nSample embedding structure for {sample_id}:")
        print(f"  personal: shape {sample['personal'].shape}")
        print(f"  org: shape {sample['org'].shape}")
        print(f"  intent: shape {sample['intent'].shape}")

    print("+" * 60)
    print("Test 2: Assymetric similarity")
    print("+" * 60)
    if len(embeddings_map) >= 2:
        user_a_id = list(embeddings_map.keys())[0]
        user_b_id = list(embeddings_map.keys())[1]

        user_a_emb = embeddings_map[user_a_id]
        user_b_emb = embeddings_map[user_b_id]

        scores = service.calculate_assymetric_similarity(user_a_emb, user_b_emb)
        print(f"\nAsymmetric similarity between {user_a_id} and {user_b_id}:")
        print(f"  A→B (A wants B): {scores['a_to_b']:.4f}")
        print(f"  B→A (B wants A): {scores['b_to_a']:.4f}")
        print(f"  Personal sim: {scores['personal']:.4f}")
        print(f"  Org sim: {scores['org']:.4f}")
        print(f"  Bidirectional: {scores['bidirectional']:.4f}")

    print("+" * 60)
    print("Test 3: Vectorized Calculation for all candidates")
    print("+" * 60)
    test_user_id = user_ids[0]
    print(f"Test user: {test_user_id}")
    compatible = await filter_service.filter_compatible_candidates(test_user_id)
    print(f"Compatible candidates: {len(compatible)}")

    matches = await service.calculate_matches_vectorized(test_user_id, compatible)

    print(f"\nCalculated {len(matches)} matches")
    print("\nTop 5 matches:")
    for i, match in enumerate(matches[:5], 1):
        print(
            f"{i}. {match['matched_user_id']} - Score: {match['similarity_score']:.4f}/{match['percentage_score']}% \n {match['similarity_breakdown']}"
        )


if __name__ == "__main__":
    import asyncio

    from src.database import connect_to_mongo, get_database
    from src.recommendations.user_filter_service import UserFilterService

    asyncio.run(test())
