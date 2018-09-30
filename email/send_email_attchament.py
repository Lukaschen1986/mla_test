import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEBase, MIMEMultipart
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
# mail
mail = MIMEMultipart()

text = "Hello World."
msg = MIMEText("<html><hl>" + text + "</hl></html>", "html", "utf-8")
msg["From"] = fr
msg["To"] = to
msg["Subject"] = sbjt
mail.attach(msg) # 附上文本

with open("file.html", "rb") as f:
    attachment= f.read()

attachment = MIMEBase(attachment, "base64", "utf-8")
attachment["Content-Type"] = "application/octet-stream"
attachment["Content-Disposition"] = "attachment; filename='file.html'"
mail.attach(attachment) # 附上附件
# send email
try:
    smtp = smtplib.SMTP_SSL(host=server.encode(), port=465) # 加密，465为默认端口
    smtp.login(user, password)
    smtp.sendmail(sender, [receiver], mail.as_string())
    smtp.quit()
except Exception as e:
    print(e)
