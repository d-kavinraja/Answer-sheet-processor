import smtplib
import random
import string
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging

logger = logging.getLogger(__name__)

class EmailService:
    """Handles sending emails for OTP verification and user communication."""
    def __init__(self, smtp_server: str, smtp_port: int, email_user: str, email_password: str):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.email_user = email_user
        self.email_password = email_password

    def generate_otp(self) -> str:
        """Generate a 6-digit numeric OTP."""
        return "".join(random.choices(string.digits, k=6))

    def send_otp(self, recipient_email: str, otp: str, username: str):
        """Sends a formatted OTP email to the recipient."""
        msg = MIMEMultipart()
        msg['From'] = self.email_user
        msg['To'] = recipient_email
        msg['Subject'] = "Smart Answer Sheet Scanner - Verify Your Email"
        body = f"""
        <html><body><div style="font-family: Arial, sans-serif; max-width: 600px; margin: auto; padding: 20px; border: 1px solid #ddd; border-radius: 10px;">
        <h2 style="color: #2E86AB; text-align: center;">📝 Smart Answer Sheet Scanner</h2>
        <h3 style="color: #333;">Email Verification Required</h3>
        <p>Hello <strong>{username}</strong>,</p>
        <p>Thank you for signing up! To secure your account, please use the following One-Time Password (OTP):</p>
        <div style="background-color: #f8f9fa; padding: 20px; text-align: center; margin: 20px 0;"><h2 style="color: #2E86AB; font-size: 32px; letter-spacing: 5px; margin: 0;">{otp}</h2></div>
        <p>This OTP is valid for 10 minutes. If you did not request this, please ignore this email.</p>
        </div></body></html>
        """
        msg.attach(MIMEText(body, 'html'))
        try:
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.email_user, self.email_password)
                server.send_message(msg)
        except smtplib.SMTPAuthenticationError as e:
            logger.error(f"SMTP Auth Error: {e}. Check credentials/App Password.")
            raise Exception("Failed to send OTP due to an authentication error.")
        except Exception as e:
            logger.error(f"General error sending OTP: {e}")
            raise Exception(f"Failed to send OTP: {e}")

    def send_welcome_email(self, recipient_email: str, username: str):
        """Send a welcome email after successful verification."""
        msg = MIMEMultipart()
        msg['From'] = self.email_user
        msg['To'] = recipient_email
        msg['Subject'] = "Welcome to the Smart Answer Sheet Scanner!"
        body = f"""
        <html><body><div style="font-family: Arial, sans-serif; max-width: 600px; margin: auto; padding: 20px; border: 1px solid #ddd; border-radius: 10px;">
        <h2 style="color: #2E86AB; text-align: center;">🎉 Welcome, {username}!</h2>
        <p>Your account is verified. You can now use the <strong>Smart Answer Sheet Scanner</strong> to process answer sheets, capture them via camera, and view your scan history.</p>
        </div></body></html>
        """
        msg.attach(MIMEText(body, 'html'))
        try:
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.email_user, self.email_password)
                server.send_message(msg)
        except Exception as e:
            logger.error(f"Failed to send welcome email to {recipient_email}: {e}")
