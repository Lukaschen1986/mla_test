# -*- coding: utf-8 -*-
import requests
from bs4 import BeautifulSoup
import random
import pandas as pd
import pickle

total_url = "http://data.weather.gov.hk/gts/time/conversion1_text_c.htm"

user_agent = ['Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.87 Safari/537.36',  
              'Mozilla/5.0 (X11; U; Linux x86_64; zh-CN; rv:1.9.2.10) Gecko/20100922 Ubuntu/10.10 (maverick) Firefox/3.6.10',  
              'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',  
              'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/30.0.1599.101 Safari/537.36',  
              'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.1 (KHTML, like Gecko) Chrome/21.0.1180.71 Safari/537.1 LBBROWSER']

headers = {'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8', 
           'Accept-Encoding': 'gzip, deflate, sdch', 
           'Accept-Language': 'zh-CN,zh;q=0.8', 
           'User-Agent': user_agent[random.randint(0,5)]}

total_request = requests.get(total_url, headers = headers)
total_text = total_request.text
total_soup = BeautifulSoup(total_text, "html.parser")

df_calender = pd.DataFrame()
for href in total_soup.find("dl").find_all("a"):
    path = href.attrs["href"]
    year_check = int(path[23:27])
    if year_check < 2013 or year_check >= 2021: # [2013,2020]
        continue
    else:
        url = "http://data.weather.gov.hk/gts" + path[2:]
        sub_request = requests.get(url, headers = headers)
        sub_request.encoding = sub_request.apparent_encoding
        text = sub_request.text
        text_new = text.split("\r\n")[3:]
        solar_date_res = []; lunar_date_res = []; solar_term_res = []
        for data in text_new:
            solar_date = pd.Timestamp(data[0:11].replace("年", "-").replace("月", "-").replace("日", "").replace(" ", ""))
            lunar_date = data[16:25].replace(" ", "")
            solar_term = data[35:44].replace(" ", "")
            solar_date_res.append(solar_date)
            lunar_date_res.append(lunar_date)
            solar_term_res.append(solar_term)
            df_year = pd.DataFrame({"solar_date": solar_date_res, 
                                    "lunar_date": lunar_date_res,
                                    "solar_term": solar_term_res}, columns=["solar_date","lunar_date","solar_term"])
            df_year = df_year.dropna(how="any", axis=0)
        df_calender = pd.concat([df_calender, df_year], axis=0)

f = open("df_calender.txt", "wb")
pickle.dump(df_calender, f)
f.close()
