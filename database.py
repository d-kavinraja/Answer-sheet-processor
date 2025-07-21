from pymongo import MongoClient
import bcrypt
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class MongoManager:
    def __init__(self, mongo_uri: str):
        """Initialize MongoDB connection."""
        logger.debug("Initializing MongoManager")
        self.client = MongoClient(mongo_uri)
        self.db = self.client["answer_sheet_scanner"]
        self.users = self.db["users"]
        self.scans = self.db["scans"]

    def create_user(self, user_data: dict) -> bool:
        """Create a new user with hashed password."""
        logger.debug(f"Creating user: {user_data['username']}")
        try:
            # Check if username or email already exists
            if self.users.find_one({"$or": [{"username": user_data["username"]}, {"email": user_data["email"]}]}):
                return False
            # Hash password
            hashed_password = bcrypt.hashpw(user_data["password"].encode("utf-8"), bcrypt.gensalt())
            user_data["password"] = hashed_password
            user_data["created_at"] = datetime.utcnow()
            self.users.insert_one(user_data)
            logger.debug("User created successfully")
            return True
        except Exception as e:
            logger.error(f"Error creating user: {str(e)}")
            return False

    def verify_user(self, email: str, password: str) -> dict:
        """Verify user credentials and return user data."""
        logger.debug(f"Verifying user: {email}")
        try:
            user = self.users.find_one({"email": email})
            if user and bcrypt.checkpw(password.encode("utf-8"), user["password"]):
                logger.debug("User verified successfully")
                return user
            logger.warning("Invalid email or password")
            return None
        except Exception as e:
            logger.error(f"Error verifying user: {str(e)}")
            return None

    def save_otp(self, email: str, otp: str):
        """Save OTP for email verification."""
        logger.debug(f"Saving OTP for {email}")
        try:
            self.users.update_one(
                {"email": email},
                {"$set": {"otp": otp, "otp_timestamp": datetime.utcnow()}},
                upsert=True
            )
            logger.debug("OTP saved successfully")
        except Exception as e:
            logger.error(f"Error saving OTP: {str(e)}")

    def verify_otp(self, email: str, otp: str) -> bool:
        """Verify OTP and mark user as verified."""
        logger.debug(f"Verifying OTP for {email}")
        try:
            user = self.users.find_one({"email": email, "otp": otp})
            if user:
                self.users.update_one(
                    {"email": email},
                    {"$set": {"verified": True}, "$unset": {"otp": "", "otp_timestamp": ""}}
                )
                logger.debug("OTP verified, user marked as verified")
                return True
            logger.warning("Invalid OTP")
            return False
        except Exception as e:
            logger.error(f"Error verifying OTP: {str(e)}")
            return False

    def save_scan(self, email: str, image_path: str, results: list):
        """Save scan results to the database."""
        logger.debug(f"Saving scan for {email}")
        try:
            scan_data = {
                "email": email,
                "image_path": image_path,
                "results": results,
                "timestamp": datetime.utcnow()
            }
            self.scans.insert_one(scan_data)
            logger.debug("Scan saved successfully")
        except Exception as e:
            logger.error(f"Error saving scan: {str(e)}")

    def get_user_scans(self, email: str) -> list:
        """Retrieve all scans for a user."""
        logger.debug(f"Retrieving scans for {email}")
        try:
            return list(self.scans.find({"email": email}).sort("timestamp", -1))
        except Exception as e:
            logger.error(f"Error retrieving scans: {str(e)}")
            return []
