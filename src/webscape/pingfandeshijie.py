import requests
from bs4 import BeautifulSoup

headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:124.0) Gecko/20100101 Firefox/124.0"}
base_url = "https://www.pingfandeshijie.cn/"  # 首页
# 三部曲
first_part = base_url + "diyibu/"  # 共有 54 章, 序号 1 -> 54，1, 1_2，1_3 分别为第一章的第一、第二、第三小节; 下同
second_part = base_url + "dierbu/"  # 共有 54 章, 序号 55 -> 108
third_part = base_url + "disanbu/"  # 共有 54 章, 序号 109 -> 162

chap01 = requests.get(first_part + "1_2.html", headers=headers)

# Create a BeautifulSoup object
soup = BeautifulSoup(chap01.text, 'html.parser')
#  print(soup.select_one(".span12 p:nth-of-type(1)").text)  # works; 2024-04-07 Sun

print(soup.select_one(".span12").text)  # works; 2024-04-07 Sun
