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


目标：并行爬取章节内容

思路：

01 根据网页组织的规则，构建url池，
    (所有章的第一节的url池, 所有章的第一节的url池)
    (url_partX_chapN_1, url_partX_chapN_b), b > 1

02 获取 <div class=pagination> 的第二个 <b></b> 可得到此章的最大节数目(m)

03 根据 m 分别构造此章第一节和非第一节的url:
    - base_url + partX + n.html
    - base_url + partX + n_m.html

04 async/await + multi-processing
"""

import requests
from bs4 import BeautifulSoup

headers = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:124.0) Gecko/20100101 Firefox/124.0"
}
base_url = "https://www.pingfandeshijie.cn/"  # 首页


if __name__ == "__main__":
    # 三部曲, 每部共有 54 章,
    #  1, 1_2，1_3 分别为第一章的第一、第二、第三小节; 下同
    first_part = base_url + "diyibu/"   # 共有 54 章, 序号 1 -> 54
    second_part = base_url + "dierbu/"  # 共有 54 章, 序号 55 -> 108
    third_part = base_url + "disanbu/"  # 共有 54 章, 序号 109 -> 162

    def demo1():
        part1_chap1 = requests.get(first_part + "1.html", headers=headers)

        # Create a BeautifulSoup object
        soup = BeautifulSoup(part1_chap1.text, "html.parser")

        print(soup.select_one(".span12 p:nth-of-type(1)").text)  # works; 2024-04-07 Sun


    def demo2():
        part1_chap1_2 = requests.get(first_part + "1_2.html", headers=headers)

        # Create a BeautifulSoup object
        soup = BeautifulSoup(part1_chap1_2.text, "html.parser")

        print(soup.select_one(".span12").text)  # works; 2024-04-07 Sun


    demo1()
    demo2()
