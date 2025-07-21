import streamlit as st
import pymongo
import bcrypt
import datetime
from datetime import timedelta
from typing import List, Dict

class MongoManager:
    def __init__(self, uri: str):
        try:
            self.client = pymongo.MongoClient(uri)
            self.client.admin.command('ping')
            self.db = self.client["smart_answer_sheet_scanner"]
            self.users_collection = self.db["users"]
            self.otp_collection = self.db["otp_verifications"]
            self.scans_collection = self.db["scans"]
            self.users_collection.create_index("username", unique=True)
            self.users_collection.create_index("email", unique=True)
            self.otp_collection.create_index("expires_at", expireAfterSeconds=0)
            self.scans_collection.create_index([("username", pymongo.ASCENDING), ("timestamp", pymongo.DESCENDING)])
        except pymongo.errors.ConnectionFailure as e:
            st.error(f"Database connection failed: {e}", icon="🚨")
            st.stop()
        except Exception as e:
            st.error(f"An error occurred with the database setup: {e}", icon="🚨")
            st.stop()

    def _hash_password(self, password: str) -> bytes:
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

    def check_password(self, password: str, hashed: bytes) -> bool:
        return bcrypt.checkpw(password.encode('utf-8'), hashed)

    def add_user(self, username: str, email: str, password: str) -> Dict:
        if self.users_collection.find_one({"username": username}):
            return {"success": False, "message": "Username already exists"}
        if self.users_collection.find_one({"email": email}):
            return {"success": False, "message": "Email already registered"}
        hashed_password = self._hash_password(password)
        user_data = {
            "username": username,
            "email": email,
            "password_hash": hashed_password,
            "email_verified": False,
            "created_at": datetime.datetime.now(datetime.UTC),
            "last_login": None
        }
        self.users_collection.insert_one(user_data)
        return {"success": True, "message": "User created successfully"}

    def find_user(self, username: str) -> Dict | None:
        return self.users_collection.find_one({"username": username})

    def find_user_by_email(self, email: str) -> Dict | None:
        return self.users_collection.find_one({"email": email})

    def store_otp(self, email: str, otp: str) -> bool:
        try:
            self.otp_collection.delete_many({"email": email})
            otp_data = {
                "email": email,
                "otp": otp,
                "created_at": datetime.datetime.now(datetime.UTC),
                "expires_at": datetime.datetime.now(datetime.UTC) + timedelta(minutes=10),
                "attempts": 0
            }
            self.otp_collection.insert_one(otp_data)
            return True
        except Exception as e:
            st.error(f"Failed to store OTP: {str(e)}")
            return False

    def verify_otp(self, email: str, otp: str) -> Dict:
        try:
            otp_record = self.otp_collection.find_one({"email": email})
            if not otp_record:
                return {"success": False, "message": "No OTP found for this email"}
            if datetime.datetime.now(datetime.UTC) > otp_record["expires_at"].replace(tzinfo=datetime.UTC):
                self.otp_collection.delete_one({"email": email})
                return {"success": False, "message": "OTP has expired. Please request a new one"}
            if otp_record["attempts"] >= 3:
                self.otp_collection.delete_one({"email": email})
                return {"success": False, "message": "Too many failed attempts. Please request a new OTP"}
            if otp_record["otp"] == otp:
                self.users_collection.update_one(
                    {"email": email},
                    {"$set": {"email_verified": True}}
                )
                self.otp_collection.delete_one({"email": email})
                return {"success": True, "message": "Email verified successfully"}
            else:
                self.otp_collection.update_one(
                    {"email": email},
                    {"$inc": {"attempts": 1}}
                )
                remaining_attempts = 3 - (otp_record["attempts"] + 1)
                return {
                    "success": False,
                    "message": f"Invalid OTP. {remaining_attempts} attempts remaining"
                }
        except Exception as e:
            st.error(f"Error verifying OTP: {str(e)}")
            return {"success": False, "message": "Verification failed due to system error"}

    def update_last_login(self, username: str):
        self.users_collection.update_one(
            {"username": username},
            {"$set": {"last_login": datetime.datetime.now(datetime.UTC)}}
        )

    def store_scan_result(self, username: str, scan_data: Dict):
        try:
            scan_data["username"] = username
            scan_data["timestamp"] = datetime.datetime.now(datetime.UTC)
            self.scans_collection.insert_one(scan_data)
            return {"success": True, "message": "Scan result stored successfully"}
        except Exception as e:
            st.error(f"Failed to store scan result: {str(e)}")
            return {"success": False, "message": "Failed to store scan result"}

    def get_user_scans(self, username: str) -> List[Dict]:
        return list(self.scans_collection.find({"username": username}).sort("timestamp", pymongo.DESCENDING))