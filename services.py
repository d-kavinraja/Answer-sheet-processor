# services.py
import random
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from config import EMAIL_USER, EMAIL_PASSWORD, SMTP_SERVER, SMTP_PORT

def generate_otp(n_digits=6):
    """Generates a random n-digit OTP."""
    return "".join([str(random.randint(0, 9)) for _ in range(n_digits)])

def send_verification_email(recipient_email, otp):
    """Sends an email with the OTP to the specified recipient."""
    if not EMAIL_USER or not EMAIL_PASSWORD:
        return False, "Email service is not configured."

    message = MIMEMultipart("alternative")
    message["Subject"] = "Your Verification Code for Smart Scanner"
    message["From"] = EMAIL_USER
    message["To"] = recipient_email

    text = f"Hi,\n\nYour OTP is: {otp}\n\nThis code will expire in 10 minutes.\n\nThank you!"
    html = f"""
    <html>
    <body>
        <div style="font-family: Arial, sans-serif; text-align: center; color: #333;">
        <h2>Verification Code</h2>
        <p>Please use the following One-Time Password (OTP) to complete your action:</p>
        <p style="font-size: 24px; font-weight: bold; letter-spacing: 2px; background-color:#f0f0f0; padding: 10px 20px; border-radius: 5px; display: inline-block;">{otp}</p>
        <p>This code is valid for 10 minutes.</p>
        <hr>
        <p style="font-size: 0.9em; color: #777;">If you did not request this, please ignore this email.</p>
        </div>
    </body>
    </html>
    """
    message.attach(MIMEText(text, "plain"))
    message.attach(MIMEText(html, "html"))

    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(EMAIL_USER, EMAIL_PASSWORD)
        server.sendmail(EMAIL_USER, recipient_email, message.as_string())
        server.quit()
        return True, "Email sent successfully."
    except Exception as e:
        return False, f"Failed to send email: {e}"