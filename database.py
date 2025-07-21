from pymongo import MongoClient
import bcrypt
from datetime import datetime

class MongoManager:
    def __init__(self, mongo_uri: str):
        """Initialize MongoDB connection."""
        self.client = MongoClient(mongo_uri)
        self.db = self.client["answer_sheet_scanner"]
        self.users = self.db["users"]
        self.scans = self.db["scans"]

    def create_user(self, user_data: dict) -> bool:
        """Create a new user with hashed password."""
        try:
            # Check if username or email already exists
            if self.users.find_one({"$or": [
                {"username": user_data["username"]},
                {"email": user_data["email"]}
            ]}):
                return False

            # Hash password
            hashed_password = bcrypt.hashpw(user_data["password"].encode("utf-8"), bcrypt.gensalt())
            user_data["password"] = hashed_password
            user_data["created_at"] = datetime.utcnow()
            self.users.insert_one(user_data)
            return True
        except Exception as e:
            print(f"Error creating user: {str(e)}")
            return False

    def verify_user(self, email: str, password: str) -> dict:
        """Verify user credentials and return user data."""
        try:
            user = self.users.find_one({"email": email})
            if user and bcrypt.checkpw(password.encode("utf-8"), user["password"]):
                return user
            return None
        except Exception as e:
            print(f"Error verifying user: {str(e)}")
            return None

    def save_otp(self, email: str, otp: str):
        """Save OTP for email verification."""
        try:
            self.users.update_one(
                {"email": email},
                {"$set": {"otp": otp, "otp_timestamp": datetime.utcnow()}},
                upsert=True
            )
        except Exception as e:
            print(f"Error saving OTP: {str(e)}")

    def verify_otp(self, email: str, otp: str) -> bool:
        """Verify OTP and mark user as verified."""
        try:
            user = self.users.find_one({"email": email, "otp": otp})
            if user:
                self.users.update_one(
                    {"email": email},
                    {"$set": {"verified": True}, "$unset": {"otp": "", "otp_timestamp": ""}}
                )
                return True
            return False
        except Exception as e:
            print(f"Error verifying OTP: {str(e)}")
            return False

    def save_scan(self, email: str, image_path: str, results: list):
        """Save scan results to the database."""
        try:
            scan_data = {
                "email": email,
                "image_path": image_path,
                "results": results,
                "timestamp": datetime.utcnow()
            }
            self.scans.insert_one(scan_data)
        except Exception as e:
            print(f"Error saving scan: {str(e)}")

    def get_user_scans(self, email: str) -> list:
        """Retrieve all scans for a user."""
        try:
            return list(self.scans.find({"email": email}).sort("timestamp", -1))
        except Exception as e:
            print(f"Error retrieving scans: {str(e)}")
            return []
