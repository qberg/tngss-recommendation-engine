@router.get("/api/profile-options/{profile_type}")
async def get_profile_options(profile_type: ProfileType):
    return {
        "profile_type": profile_type,
        "offerings": {
            "applicable": profile_type
            not in ["mentor_sme", "aspirants_individuals", "others"],
            "options": (
                OFFERINGS.get(profile_type, [])
                if profile_type not in ["mentor_sme", "aspirants_individuals", "others"]
                else []
            ),
        },
        "looking_for": {
            "applicable": True,  # All profiles have this
            "options": LOOKING_FOR.get(profile_type, []),
        },
    }


async def initialize_indexes():
    """
    Create all necessary indexes for the recommendation system.
    Called automatically on startup. Idempotent - safe to run multiple times.
    """
    try:
        db = mongodb.get_database()
        logger.info("[***] Initializing database indexes...")

        await db[settings.USERS_PROFILE_COLLECTION].create_index(
            [("user_id", ASCENDING)], name="idx_user_profile_user_id", unique=True
        )
        logger.info(
            f"[SUCCESS] Index created on {settings.USERS_PROFILE_COLLECTION}.user_id"
        )

        await db[settings.CONTEXT_BUILDER_COLLECTION].create_index(
            [("user_id", ASCENDING)], name="idx_context_user_id", unique=True
        )
        logger.info(
            f"[SUCCESS] Index created on {settings.CONTEXT_BUILDER_COLLECTION}.user_id"
        )

        await db[settings.RECOMMENDATIONS_COLLECTION].create_index(
            [("user_id", ASCENDING), ("updated_at", ASCENDING)],
            name="idx_recs_user_updated",
        )

        await db[settings.RECOMMENDATIONS_COLLECTION].create_index(
            [("user_id", ASCENDING), ("score", ASCENDING)], name="idx_recs_user_score"
        )

        recommendation_indexes = [
            IndexModel(
                [
                    ("user_id", ASCENDING),
                    ("reference_type", ASCENDING),
                    ("updated_at", DESCENDING),
                ],
                name="user_ref_type_updated_idx",
            ),
            IndexModel(
                [
                    ("user_id", ASCENDING),
                    ("reference_type", ASCENDING),
                    ("score", DESCENDING),
                ],
                name="user_ref_type_score_idx",
            ),
            IndexModel(
                [
                    ("user_id", ASCENDING),
                    ("reference_id", ASCENDING),
                    ("reference_type", ASCENDING),
                ],
                name="user_ref_unique_idx",
                unique=True,
            ),
        ]

        await db[settings.RECOMMENDATIONS_COLLECTION].create_indexes(
            recommendation_indexes
        )
        logger.info(
            f"[SUCCESS] Indexes created on {settings.RECOMMENDATIONS_COLLECTION}"
        )
        for collection_name in [
            settings.USERS_PROFILE_COLLECTION,
            settings.CONTEXT_BUILDER_COLLECTION,
            settings.RECOMMENDATIONS_COLLECTION,
        ]:
            indexes = await db[collection_name].index_information()
            logger.info(f"[INDEXES] {collection_name}: {list(indexes.keys())}")

        logger.info("[SUCCESS] All indexes initialized")
        return True

    except Exception as e:
        logger.error(f"[FAILED] Error initializing indexes: {e}")
        return False
