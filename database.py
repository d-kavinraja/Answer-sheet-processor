# database.py
import pymongo
from datetime import datetime, timedelta
from config import MONGO_URI, OTP_VALIDITY_MINUTES, DB_NAME
from models import hash_password, verify_password

def get_db_connection():
    """Establishes a connection to MongoDB and returns the database object."""
    try:
        client = pymongo.MongoClient(MONGO_URI)
        # Ping to confirm connection
        client.admin.command('ping')
        return client[DB_NAME]
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        return None

def find_user_by_email(email):
    """Finds a user by their email address."""
    db = get_db_connection()
    if db is None: return None
    return db.users.find_one({"email": email})

def add_user(email, password, otp):
    """Adds a new, unverified user to the database."""
    db = get_db_connection()
    if db is None: return False
    
    salt, hashed_pw = hash_password(password)
    expiry_time = datetime.now() + timedelta(minutes=OTP_VALIDITY_MINUTES)
    
    try:
        db.users.insert_one({
            "email": email,
            "salt": salt,
            "password_hash": hashed_pw,
            "is_verified": False,
            "otp": otp,
            "otp_expiry": expiry_time
        })
        return True
    except pymongo.errors.DuplicateKeyError:
        return False

def update_otp_for_user(email, otp):
    """Updates the OTP for an existing user."""
    db = get_db_connection()
    if db is None: return
    
    expiry_time = datetime.now() + timedelta(minutes=OTP_VALIDITY_MINUTES)
    db.users.update_one(
        {"email": email},
        {"$set": {"otp": otp, "otp_expiry": expiry_time}}
    )

def verify_user_otp(email, otp):
    """Verifies a user's OTP and marks them as verified."""
    db = get_db_connection()
    if db is None: return False
    
    user = db.users.find_one({"email": email})
    if user and user['otp'] == otp and datetime.now() < user['otp_expiry']:
        db.users.update_one(
            {"email": email},
            {"$set": {"is_verified": True, "otp": None, "otp_expiry": None}}
        )
        return True
    return False