import smtplib
import random
import string
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

class EmailService:
    def __init__(self, smtp_server: str, smtp_port: int, email_user: str, email_password: str):
        """Initialize the email service with SMTP credentials."""
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.email_user = email_user
        self.email_password = email_password

    def generate_otp(self) -> str:
        """Generate a 6-digit OTP."""
        return ''.join(random.choices(string.digits, k=6))

    def send_otp(self, recipient_email: str, otp: str):
        """Send an OTP to the recipient's email."""
        try:
            # Set up the MIME
            message = MIMEMultipart()
            message["From"] = self.email_user
            message["To"] = recipient_email
            message["Subject"] = "Your OTP for Smart Answer Sheet Scanner"
            
            body = f"Your OTP for email verification is: {otp}\n\nThis OTP is valid for 10 minutes."
            message.attach(MIMEText(body, "plain"))

            # Connect to SMTP server
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.email_user, self.email_password)
                server.sendmail(self.email_user, recipient_email, message.as_string())
        except Exception as e:
            print(f"Error sending OTP: {str(e)}")
            raise Exception(f"Failed to send OTP: {str(e)}")
