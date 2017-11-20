import smtplib
from email.mime.text import MIMEText
from email.header import Header

def python_email(receiver, subject, html_text):
    server = "smtp.huawei.com"
    sender = "itravel@huawei.com"
    username = "pub12277"
    password = "Hwsl@201711"
    msg = MIMEText("<html><hl>"+html_text+"</hl></html>", "html", "utf-8")
    msg["Subject"] = subject
    smtp = smtplib.SMTP()
    smtp.connect(server)
    smtp.login(username, password)
    smtp.sendmail(sender, receiver, msg.as_string())
    return smtp.quit()
