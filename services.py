import streamlit as st
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import random

class EmailService:
    def __init__(self, smtp_server: str, smtp_port: int, email_user: str, email_password: str):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.email_user = email_user
        self.email_password = email_password
    
    def generate_otp(self) -> str:
        return str(random.randint(100000, 999999))
    
    def send_otp_email(self, recipient_email: str, otp: str, username: str) -> bool:
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_user
            msg['To'] = recipient_email
            msg['Subject'] = "Smart Answer Sheet Scanner - Email Verification"
            body = f"""
            <html>
                <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
                    <div style="max-width: 600px; margin: 0 auto; padding: 20px; border: 1px solid #ddd; border-radius: 10px;">
                        <h2 style="color: #2E86AB; text-align: center;">📝 Smart Answer Sheet Scanner</h2>
                        <h3 style="color: #333;">Email Verification Required</h3>
                        <p>Hello <strong>{username}</strong>,</p>
                        <p>Thank you for signing up! Please verify your email address using the OTP below:</p>
                        <div style="background-color: #f8f9fa; padding: 20px; border-radius: 5px; text-align: center; margin: 20px 0;">
                            <h2 style="color: #2E86AB; font-size: 32px; letter-spacing: 5px; margin: 0;">{otp}</h2>
                        </div>
                        <p><strong>Important:</strong> This OTP will expire in 10 minutes.</p>
                        <p>If you didn't create an account, please ignore this email.</p>
                        <hr style="margin: 30px 0; border: none; border-top: 1px solid #eee;">
                        <p style="font-size: 12px; color: #666; text-align: center;">
                            This is an automated message from Smart Answer Sheet Scanner.<br>
                            Please do not reply to this email.
                        </p>
                    </div>
                </body>
            </html>
            """
            msg.attach(MIMEText(body, 'html'))
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.email_user, self.email_password)
                server.send_message(msg)
            return True
        except Exception as e:
            st.error(f"Failed to send email: {str(e)}")
            return False
    
    def send_welcome_email(self, recipient_email: str, username: str) -> bool:
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_user
            msg['To'] = recipient_email
            msg['Subject'] = "Welcome to Smart Answer Sheet Scanner!"
            body = f"""
            <html>
                <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
                    <div style="max-width: 600px; margin: 0 auto; padding: 20px; border: 1px solid #ddd; border-radius: 10px;">
                        <h2 style="color: #2E86AB; text-align: center;">🎉 Welcome to Smart Answer Sheet Scanner!</h2>
                        <p>Hello <strong>{username}</strong>,</p>
                        <p>Your email has been verified, and your account is now active.</p>
                        <div style="background-color: #d4edda; padding: 15px; border-radius: 5px; border-left: 4px solid #28a745; margin: 20px 0;">
                            <h4 style="margin: 0; color: #155724;">🚀 Ready to Get Started?</h4>
                            <p style="margin: 5px 0 0 0; color: #155724;">
                                Access powerful features to scan and extract data from answer sheets:
                            </p>
                            <ul style="color: #155724; margin: 10px 0 0 20px;">
                                <li>Automatic detection of register numbers and subject codes</li>
                                <li>Real-time camera scanning</li>
                                <li>PDF and image processing</li>
                                <li>Secure scan history storage</li>
                            </ul>
                        </div>
                        <p>Start scanning your answer sheets today!</p>
                        <hr style="margin: 30px 0; border: none; border-top: 1px solid #eee;">
                        <p style="font-size: 12px; color: #666; text-align: center;">
                            Thank you for choosing Smart Answer Sheet Scanner.<br>
                            If you have any questions, reach out to our support team.
                        </p>
                    </div>
                </body>
            </html>
            """
            msg.attach(MIMEText(body, 'html'))
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.email_user, self.email_password)
                server.send_message(msg)
            return True
        except Exception as e:
            st.error(f"Failed to send welcome email: {str(e)}")
            return False