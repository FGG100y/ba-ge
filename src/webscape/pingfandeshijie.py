""" 路遥：《平凡的世界》三部曲，网络爬虫

"https://www.pingfandeshijie.cn/"

<body>
    <div class=container>
        <div class=row>
            <div class=span12>

                <p>
                    第n章第一节的正文内容...
                    <!-- n 为本部的某一章，整数 --> 
                </p>

                <div class=pagenation>
                    <ul>
                        <a title=page>
                            <b>1</b>
                            /
                            <b>m</b>
                            <!-- m 为本章最后一节，整数 --> 
                        </a>

<body>
    <div class=container>
        <div class=row>
            <div class=span12>

                第n章"非"第一节的正文内容...
                <!-- n 为本部的某一章，整数 ; 注意：没有 <p></p> --> 
                <!-- 注意2: <p></p> 标签内容是 “上一篇 x，下一篇 y”的链接 -->

                <div class=pagenation>
                    <ul>
                        <a title=page>
                            <b>1</b>
                            /
                            <b>m</b>
                            <!-- m 为本章最后一节，整数 --> 
                        </a>
"""
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
