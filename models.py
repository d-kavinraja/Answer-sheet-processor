# models.py
import hashlib
import os
from config import HASH_SECRET_KEY

def hash_password(password, salt=None):
    """Hashes a password with a salt using PBKDF2."""
    if salt is None:
        salt = os.urandom(16)  # Generate a new random salt
    
    # Use PBKDF2 for password hashing
    hashed_password = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt,
        100000, # Number of iterations
        dklen=128
    )
    # Return salt and hash, so we can store both
    return salt, hashed_password

def verify_password(stored_salt, stored_hash, provided_password):
    """Verifies a provided password against a stored salt and hash."""
    # Hash the provided password with the stored salt
    _, hashed_password_to_check = hash_password(provided_password, stored_salt)
    return stored_hash == hashed_password_to_check

# User Schema for MongoDB:
# {
#     "email": TEXT (unique),
#     "salt": BINARY,
#     "password_hash": BINARY,
#     "is_verified": BOOLEAN,
#     "otp": TEXT,
#     "otp_expiry": DATETIME
# }