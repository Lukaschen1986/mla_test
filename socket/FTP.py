import ftplib

HOST = "ftp.acc.umu.se"
DIR = "public/EFLIB"
FILE = "README"

f = ftplib.FTP()
f.set_debuglevel(2)
f.connect(HOST)
f.login()
f.cwd(DIR)
f.retrbinary("retr {}".format(FILE), open(FILE, "wb").write)
f.quit()
