import smtplib
from email.mime.text import MIMEText
from email.header import Header

# server
server = "smtp.qq.com"
# sender
sender = "438320628@qq.com"
fr = Header(sender, "utf-8")
#sender_code = ""
user = "438320628"
password = ""
# receiver
receiver = "chen_wen_dong@126.com"
to = Header(receiver, "utf-8")
# subject
subject = "this is subject"
sbjt = Header(subject, "utf-8")
# msg
text = "Hello World."
#msg = MIMEText(text, "plain", "utf-8")
msg = MIMEText("<html><hl>" + text + "</hl></html>", "html", "utf-8")
msg["From"] = fr
msg["To"] = to
msg["Subject"] = sbjt
# send email
try:
    #smtp = smtplib.SMTP() # 不加密，不安全
    #smtp.connect(server)
    smtp = smtplib.SMTP_SSL(host=server.encode(), port=465) # 加密，465为默认端口
    smtp.login(user, password)
    #smtp.login(sender, sender_code)
    smtp.sendmail(sender, [receiver], msg.as_string())
    smtp.quit()
except Exception as e:
    print(e)
