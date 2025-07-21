# services.py

import streamlit as st
import smtplib
import random
import string
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging

logger = logging.getLogger(__name__)

class EmailService:
    """
    Handles sending emails for OTP verification and user communication.
    Uses HTML templates for a better user experience.
    """
    def __init__(self, smtp_server: str, smtp_port: int, email_user: str, email_password: str):
        """Initialize the email service with SMTP credentials."""
        logger.debug("Initializing EmailService")
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.email_user = email_user
        self.email_password = email_password

    def generate_otp(self) -> str:
        """Generate a 6-digit numeric OTP."""
        logger.debug("Generating OTP")
        return "".join(random.choices(string.digits, k=6))

    def send_otp(self, recipient_email: str, otp: str, username: str):
        """Send a beautifully formatted OTP email to the recipient."""
        logger.debug(f"Sending OTP to {recipient_email} for user {username}")
        
        msg = MIMEMultipart()
        msg['From'] = self.email_user
        msg['To'] = recipient_email
        msg['Subject'] = "Smart Answer Sheet Scanner - Verify Your Email"
        
        # Use a professional HTML template for the email body
        body = f"""
        <html>
            <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
                <div style="max-width: 600px; margin: 0 auto; padding: 20px; border: 1px solid #ddd; border-radius: 10px;">
                    <h2 style="color: #2E86AB; text-align: center;">📝 Smart Answer Sheet Scanner</h2>
                    <h3 style="color: #333;">Email Verification Required</h3>
                    <p>Hello <strong>{username}</strong>,</p>
                    <p>Thank you for signing up! To secure your account, please use the following One-Time Password (OTP):</p>
                    <div style="background-color: #f8f9fa; padding: 20px; border-radius: 5px; text-align: center; margin: 20px 0;">
                        <h2 style="color: #2E86AB; font-size: 32px; letter-spacing: 5px; margin: 0;">{otp}</h2>
                    </div>
                    <p>This OTP is valid for 10 minutes. If you did not request this, please ignore this email.</p>
                    <hr style="margin: 30px 0; border: none; border-top: 1px solid #eee;">
                    <p style="font-size: 12px; color: #666; text-align: center;">This is an automated message. Please do not reply.</p>
                </div>
            </body>
        </html>
        """
        msg.attach(MIMEText(body, 'html'))
        
        try:
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.email_user, self.email_password)
                server.send_message(msg)
            logger.debug(f"OTP email sent successfully to {recipient_email}")
        except Exception as e:
            logger.error(f"Failed to send OTP email to {recipient_email}: {str(e)}")
            # Raise the exception to be caught by the UI layer
            raise Exception(f"Failed to send OTP: {str(e)}")

    def send_welcome_email(self, recipient_email: str, username: str):
        """Send a welcome email after successful verification."""
        logger.debug(f"Sending welcome email to {recipient_email}")

        msg = MIMEMultipart()
        msg['From'] = self.email_user
        msg['To'] = recipient_email
        msg['Subject'] = "Welcome to the Smart Answer Sheet Scanner!"

        body = f"""
        <html>
            <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
                <div style="max-width: 600px; margin: 0 auto; padding: 20px; border: 1px solid #ddd; border-radius: 10px;">
                    <h2 style="color: #2E86AB; text-align: center;">🎉 Welcome, {username}!</h2>
                    <p>Your account has been successfully verified. You are now ready to use the <strong>Smart Answer Sheet Scanner</strong>.</p>
                    <div style="background-color: #d4edda; padding: 15px; border-radius: 5px; border-left: 4px solid #28a745; margin: 20px 0;">
                        <h4 style="margin: 0; color: #155724;">🚀 You can now:</h4>
                        <ul style="color: #155724; margin: 10px 0 0 20px;">
                            <li>Upload answer sheets for automatic processing.</li>
                            <li>Use your camera to capture and scan sheets in real-time.</li>
                            <li>View your complete scan history.</li>
                        </ul>
                    </div>
                    <p>We're excited to have you on board!</p>
                    <hr style="margin: 30px 0; border: none; border-top: 1px solid #eee;">
                    <p style="font-size: 12px; color: #666; text-align: center;">This is an automated message. Please do not reply.</p>
                </div>
            </body>
        </html>
        """
        msg.attach(MIMEText(body, 'html'))
        
        try:
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.email_user, self.email_password)
                server.send_message(msg)
            logger.debug(f"Welcome email sent successfully to {recipient_email}")
        except Exception as e:
            # Log the error but don't block the user flow if the welcome email fails
            logger.error(f"Failed to send welcome email to {recipient_email}: {str(e)}")

