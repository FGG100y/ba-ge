"""templates for writing into mysqlDB using pymysql:
"""
import os
import random
import re
from urllib.request import urlopen

import pymysql
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()
mysql_pwd = os.getenv("mysql_password")


conn = pymysql.connect(
    host="127.0.0.1",
    #  host="localhost",
    user="dstml001",
    unix_socket="/var/run/mysqld/mysqld.sock",  # mysqladmin -u dstml001 -p variables | grep socket
    passwd=mysql_pwd,
    #  db="scraping",
    charset="utf8",
)
cur = conn.cursor()
cur.execute("USE scraping")

# NOTE connect to local mysqldb work ok; 2024-04-07 Sun

random.seed()  # a=None, the current system time is used


def store(title, content):
    cur.execute(
        "INSERT INTO pages (title, content) VALUES " '("%s", "%s")', (title, content)
    )
    cur.connection.commit()


def getLinks(articleUrl):
    html = urlopen("http://en.wikipedia.org" + articleUrl)
    bs = BeautifulSoup(html, "html.parser")
    title = bs.find("h1").get_text()
    content = bs.find("div", {"id": "mw-content-text"}).find("p").get_text()
    store(title, content)
    return bs.find("div", {"id": "bodyContent"}).find_all(
        "a", href=re.compile("^(/wiki/)((?!:).)*$")
    )


links = getLinks("/wiki/Kevin_Bacon")
try:
    while len(links) > 0:
        newArticle = links[random.randint(0, len(links) - 1)].attrs["href"]
        print(newArticle)
        links = getLinks(newArticle)
finally:
    cur.close()
    conn.close()
