from bson import ObjectId

from src.config import settings


def candidate_match_pipeline(user_id: str, user_sectors: list):
    """
    Build aggregation pipeline to find compatible candidates.
    Filters by sector overlap and joins org profile data.
    """
    user_obj_id = ObjectId(user_id)

    return [
        # Stage 1: Filter by sector overlap
        {
            "$match": {
                "user_id": {"$ne": user_obj_id},
                "sector.value": {"$in": user_sectors},
            }
        },
        # Stage 2: Projection to reduce data
        {"$project": {"user_id": 1, "looking_to_connect": 1, "_id": 0}},
        # Stage 3: Join with org column
        {
            "$lookup": {
                "from": settings.ORGANISATION_PROFILE_COLLECTION,
                "localField": "user_id",
                "foreignField": "user_id",
                "as": "org",
                "pipeline": [{"$project": {"profile_type": 1, "_id": 0}}],
            }
        },
        # Stage 4: Filter out users with org
        {"$match": {"org": {"$ne": []}}},
        # Stage 5: Unwind org array
        {"$unwind": "$org"},
        # Stage 6: Final Projection
        {
            "$project": {
                "user_id": 1,
                "looking_to_connect": 1,
                "profile_type": "$org.profile_type",
            }
        },
    ]
