from pymongo import MongoClient
import bcrypt
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class MongoManager:
    def __init__(self, mongo_uri: str):
        self.client = MongoClient(mongo_uri)
        self.db = self.client["answer_sheet_scanner"]
        self.users = self.db["users"]
        self.scans = self.db["scans"]
        self.users.create_index("email", unique=True)

    def create_user(self, user_data: dict) -> bool:
        """Create a new user with a hashed password."""
        try:
            if self.users.find_one({"email": user_data["email"]}):
                return False
            hashed_password = bcrypt.hashpw(user_data["password"].encode("utf-8"), bcrypt.gensalt())
            user_data["password"] = hashed_password
            user_data["created_at"] = datetime.utcnow()
            user_data["verified"] = False
            self.users.insert_one(user_data)
            return True
        except Exception as e:
            logger.error(f"Error creating user: {e}")
            return False

    def verify_user(self, email: str, password: str) -> dict:
        """Verify user credentials and return user data if valid."""
        user = self.users.find_one({"email": email})
        if user and bcrypt.checkpw(password.encode("utf-8"), user["password"]):
            return user
        return None

    def save_otp(self, email: str, otp: str):
        """Save OTP and its timestamp for a user."""
        self.users.update_one(
            {"email": email},
            {"$set": {"otp": otp, "otp_timestamp": datetime.utcnow()}}
        )

    def verify_otp(self, email: str, otp: str) -> dict:
        """Verify OTP. If valid and not expired, mark user as verified and return user."""
        user = self.users.find_one({"email": email, "otp": otp})
        if user:
            otp_time = user.get("otp_timestamp", datetime.min)
            if (datetime.utcnow() - otp_time) < timedelta(minutes=10):
                self.users.update_one(
                    {"email": email},
                    {"$set": {"verified": True}, "$unset": {"otp": "", "otp_timestamp": ""}}
                )
                return user
        return None

    def save_scan(self, email: str, history_item: dict):
        """Save a complete scan result (history item) to the database."""
        scan_data = {"email": email, "history_item": history_item}
        self.scans.insert_one(scan_data)

    def get_user_scans(self, email: str) -> list:
        """Retrieve all scans for a user, sorted by date."""
        return list(self.scans.find({"email": email}).sort("history_item.timestamp", -1))
