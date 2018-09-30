import re
p = r"\d+"
re.findall(string="one12twothree", pattern=p)

strg = "你好世界，Hello World"
p = r"[\u4e00-\u9fa5]+"
re.findall(string=strg, pattern=p) # ['你好世界']
