import requests
from bs4 import BeautifulSoup
import random
import xlwt
from numpy import *
from pandas import *

user_agent = ['Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.87 Safari/537.36',  
              'Mozilla/5.0 (X11; U; Linux x86_64; zh-CN; rv:1.9.2.10) Gecko/20100922 Ubuntu/10.10 (maverick) Firefox/3.6.10',  
              'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',  
              'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/30.0.1599.101 Safari/537.36',  
              'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.1 (KHTML, like Gecko) Chrome/21.0.1180.71 Safari/537.1 LBBROWSER']

headers = {'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8', 
           'Accept-Encoding': 'gzip, deflate, sdch', 
           'Accept-Language': 'zh-CN,zh;q=0.8', 
           'User-Agent': user_agent[random.randint(0,5)]} 

cities = ["beijing","shanghai","nanjing"]
first_url = "http://lishi.tianqi.com/%s/index.html" % cities[0]

index = requests.get(first_url, headers = headers) # get被选城市所有月份的天气主页
html_index  = index.text
index_soup = BeautifulSoup(html_index, "html.parser") 

df_final = DataFrame()
for href in index_soup.find("div", class_ = "tqtongji1").find_all("a"):
    # print href.attrs["href"]
    url = href.attrs["href"] # 得到每个月的url
    r = requests.get(url, headers = headers) # get
    html = r.text # text
    # print html
    soup = BeautifulSoup(html, "html.parser") # 解析月度数据
    
    row = []; rows = []
    for element in soup.find("div",class_="tqtongji2").find_all("li"):  
        # print element.string  
        row.append(element.string) # 将每一行每一列数据逐个插入row
        if len(row) == 6:
            rows.append(row)  
            row = [] # 一旦累计插入数达到6个，就形成一整行插入rows，同时row清零，再循环
    rows = rows[1:]
    df = DataFrame(rows, columns = ["日期","最高气温","最低气温","天气","风向","风力"])
    df_final = concat([df_final, df], axis = 0)
    
df_final = df_final.sort_values(by = "日期")
city_name = Series(cities[0], index = df_final.index, name = "城市")
df_final = concat([city_name, df_final], axis = 1)
