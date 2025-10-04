"""
Migration script to convert string arrays to object arrays in user profile collection.
Converts arrays like ["startup", "mentor"] to [{"label": "Startup", "value": "startup"}, ...]
"""

from src.config import settings
from src.database import get_database
from src.utils.setup_logger import setup_logger

logger = setup_logger(__name__, "logs/migration.log")


def format_label(value: str) -> str:
    """
    Convert snake_case or lowercase value to Title Case label.
    Examples:
        "startup" -> "Startup"
        "industry_corporate" -> "Industry Corporate"
        "mentor_sme" -> "Mentor Sme"
    """
    return value.replace("_", " ").title()


def convert_string_to_object(value: str) -> dict:
    """
    Convert a string value to an object with label and value.

    Args:
        value: String value like "startup" or "industry_corporate"

    Returns:
        Dict with label and value keys
    """
    return {"label": format_label(value), "value": value}


async def migrate_user_profiles():
    """
    Migrate user profile collection arrays from strings to objects.
    Fields to migrate: looking_to_connect, looking_to_meet, sector
    """
    try:
        db = get_database()
        collection = db[settings.CONTEXT_BUILDER_COLLECTION]

        # Fields that need migration
        array_fields = ["looking_to_connect", "looking_to_meet", "sector"]

        # Find all documents where at least one array field contains strings
        query = {
            "$or": [
                {field: {"$type": "array", "$exists": True}} for field in array_fields
            ]
        }

        documents = await collection.find(query).to_list(length=None)
        total_docs = len(documents)

        logger.info(f"[INFO] Found {total_docs} documents to check for migration")

        migrated_count = 0
        skipped_count = 0

        for doc in documents:
            needs_migration = False
            update_data = {}

            for field in array_fields:
                if field in doc and isinstance(doc[field], list):
                    # Check if array contains strings (old format)
                    if doc[field] and isinstance(doc[field][0], str):
                        # Convert strings to objects
                        update_data[field] = [
                            convert_string_to_object(item) for item in doc[field]
                        ]
                        needs_migration = True
                        logger.info(
                            f"[CONVERT] user_id: {doc.get('user_id')} - "
                            f"{field}: {doc[field]} -> {update_data[field]}"
                        )

            if needs_migration:
                # Update the document
                result = await collection.update_one(
                    {"_id": doc["_id"]}, {"$set": update_data}
                )

                if result.modified_count > 0:
                    migrated_count += 1
                    logger.info(
                        f"[SUCCESS] Migrated document with user_id: {doc.get('user_id')}"
                    )
                else:
                    logger.warning(
                        f"[WARNING] Failed to migrate document with user_id: {doc.get('user_id')}"
                    )
            else:
                skipped_count += 1

        logger.info(
            f"[COMPLETE] Migration complete. "
            f"Migrated: {migrated_count}, Skipped: {skipped_count}, Total: {total_docs}"
        )

        return {
            "total": total_docs,
            "migrated": migrated_count,
            "skipped": skipped_count,
        }

    except Exception as e:
        logger.error(f"[ERROR] Migration failed: {e}")
        raise


async def verify_migration():
    """
    Verify that migration was successful by checking for any remaining string arrays.
    """
    try:
        db = get_database()
        collection = db[settings.USERS_PROFILE_COLLECTION]

        array_fields = ["looking_to_connect", "looking_to_meet", "sector"]

        # Check for documents still having string arrays
        issues = []

        for field in array_fields:
            # Find documents where the field is an array and first element is a string
            query = {
                field: {"$exists": True, "$type": "array"},
                f"{field}.0": {"$type": "string"},
            }

            count = await collection.count_documents(query)

            if count > 0:
                issues.append(f"{field}: {count} documents still have string arrays")
                logger.warning(f"[ISSUE] {field}: {count} documents need migration")
            else:
                logger.info(f"[OK] {field}: All documents migrated successfully")

        if issues:
            logger.error(f"[VERIFICATION FAILED] Issues found: {issues}")
            return False
        else:
            logger.info("[VERIFICATION SUCCESS] All documents migrated successfully")
            return True

    except Exception as e:
        logger.error(f"[ERROR] Verification failed: {e}")
        return False


async def rollback_migration():
    """
    Rollback migration by converting object arrays back to string arrays.
    Use with caution!
    """
    try:
        db = get_database()
        collection = db[settings.USERS_PROFILE_COLLECTION]

        array_fields = ["looking_to_connect", "looking_to_meet", "sector"]

        # Find documents with object arrays
        query = {"$or": [{f"{field}.0": {"$type": "object"}} for field in array_fields]}

        documents = await collection.find(query).to_list(length=None)
        logger.info(f"[INFO] Found {len(documents)} documents to rollback")

        rollback_count = 0

        for doc in documents:
            update_data = {}

            for field in array_fields:
                if field in doc and isinstance(doc[field], list):
                    # Check if array contains objects (new format)
                    if doc[field] and isinstance(doc[field][0], dict):
                        # Convert objects back to strings
                        update_data[field] = [
                            item.get("value", "") for item in doc[field]
                        ]

            if update_data:
                await collection.update_one({"_id": doc["_id"]}, {"$set": update_data})
                rollback_count += 1
                logger.info(
                    f"[ROLLBACK] Rolled back document with user_id: {doc.get('user_id')}"
                )

        logger.info(
            f"[COMPLETE] Rollback complete. Rolled back {rollback_count} documents"
        )
        return rollback_count

    except Exception as e:
        logger.error(f"[ERROR] Rollback failed: {e}")
        raise


# Example usage
if __name__ == "__main__":
    import asyncio

    from src.database import close_mongo_connection, connect_to_mongo

    async def main():
        # Connect to database
        await connect_to_mongo()

        try:
            # Run migration
            logger.info("[START] Starting migration process...")
            result = await migrate_user_profiles()

            print(f"\nMigration Results:")
            print(f"  Total documents checked: {result['total']}")
            print(f"  Documents migrated: {result['migrated']}")
            print(f"  Documents skipped: {result['skipped']}")

            # Verify migration
            logger.info("\n[VERIFY] Verifying migration...")
            verification_success = await verify_migration()

            if verification_success:
                print("\n[SUCCESS] Migration completed successfully!")
            else:
                print("\n[FAILURE] Migration completed with issues. Check logs.")

        finally:
            # Close connection
            await close_mongo_connection()

    asyncio.run(main())
