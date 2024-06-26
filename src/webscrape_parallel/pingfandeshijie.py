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

                <div class=pagination>
                    <ul>
                        <a title=page>
                            <b>1</b>
                            /
                            <b>m</b>
                            <!-- m 为本章最后一节，整数 --> 
                        </a>


概念回顾：

- 并发：多个任务在同一个时间段内同时执行，单核情况下，CPU会不断切换任务来完成并发操作
- 并行：多个任务在同一个时刻同时执行，需要多核，每个核心独立执行一个任务，CPU不需切换
- 同步：多任务开始执行，任务 A, B, C 全部执行完成后才算是结束
- 异步：多任务开始执行，注需要主任务 A 执行完成就算结束。主任务执行的时候，可以执行异步
    任务 B, C，主任务 A 可以不需要等待异步任务 B, C 的结果。

并发、并行，是逻辑结构的设计模式；同步、异步，是逻辑调用方式。

并发、并行，是异步的两种实现方式。串行是同步的一种方式，所有任务一个一个执行完成。



目标：爬取章节内容；实现异步操作（并发 + 并行）

思路：

01 根据网页组织的规则，构建URLs池，三部曲就分别构建三个URLs池

02 每个URLs池分配一个核心（或者更多）进行并行处理（爬取并解析出链接页面的文字）

03 一个URLs池内的爬取任务需要保证章节顺序
    - 异步执行时，同时保存其序号（以便全部任务结束后可以排序）
    - 同步执行（串行完成所有任务, 实现简单）

04 async/await + multi-processing
"""

import joblib
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from tqdm import tqdm

import pandas as pd
from typing import IO
import sys
import logging
import asyncio
import aiofiles
import aiohttp
from aiohttp import ClientSession

logging.basicConfig(
    format="%(asctime)s %(levelname)s:%(name)s: %(message)s",
    level=logging.DEBUG,
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)
logger = logging.getLogger("acrawl")
logging.getLogger("chardet.charsetprober").disabled = True

headers = {  # F12 -> Network -> Headers -> User-Agent
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:124.0) Gecko/20100101 Firefox/124.0"
}
base_url = "https://www.pingfandeshijie.cn/"  # 首页


def get_pagination(url):
    result = requests.get(url, headers=headers)
    #  soup = BeautifulSoup(result.text, "html.parser")
    soup = BeautifulSoup(result.text, "lxml")  # parsing faster
    tot_sections = soup.select_one(".pagination b:nth-of-type(2)").text
    return tot_sections


def produce_urls():
    """Get URLs of PingFanDeShiJie trilogy"""
    part_urls = {}
    start = 1
    chaps = 54
    end = 1 + chaps
    for part in tqdm(["diyibu", "dierbu", "disanbu"]):
        urls = []
        for chap in range(start, end):
            # NOTE urljoin(base, url):
            # Use the "base" parameter with a trailing slash ("/"),
            # and avoid starting the "url" parameter with a slash ("/").
            url = urljoin(base_url, part + f"/{chap}.html")
            urls.append(url)
            sections = get_pagination(url)  # number, in string
            for sec in range(2, int(sections) + 1):
                chapi = "_".join([str(chap), str(sec)])
                url = urljoin(base_url, part + f"/{chapi}.html")
                urls.append(url)
            part_urls.update({part: urls})
            #  print(part_urls)
            #  breakpoint()
            #  break
        start += chaps
        end += chaps
    joblib.dump(part_urls, "data/pfdsj_urls_dict.pkl")
    return part_urls


async def fetch_html(url: str, session: ClientSession, **kwargs) -> str:
    """Get request wrapper to fetch page HTML.

    kwargs are passed to `session.request()`.
    """

    resp = await session.request(method="GET", url=url, **kwargs)
    resp.raise_for_status()
    #  logger.info("Got response [%s] for URL: %s", resp.status, url)
    html = await resp.text()
    return html


async def parse(url: str, session: ClientSession, **kwargs) -> set:
    """Find target texts in the HTML of `url`"""
    found = {}  # {章节编号: 内容}
    try:
        html = await fetch_html(url=url, session=session, **kwargs)
    except (
        aiohttp.ClientError,
        aiohttp.http_exceptions.HttpProcessingError,
    ) as e:
        logger.error(
            "aiohttp exception for %s [%s]: %s",
            url,
            getattr(e, "status", None),
            getattr(e, "message", None),
        )
        return found
    except Exception as e:
        logger.exception(
            "Non-aiohttp exception occured: %s", getattr(e, "__dict__", {})
        )
        return found
    else:
        key = url.split("/")[-1].split(".")[0]
        texts = BeautifulSoup(html, "lxml").select_one(".span12").text
        # type(texts) -> str;
        # get rid of unrelative short texts that just useful for web browsing:
        val = max([p for p in texts.split("\n")], key=len)
        found.update({key: val})
        return found


async def write_one(file: IO, url: str, **kwargs) -> None:
    """Write the found texts of `url` to file"""
    res = await parse(url=url, **kwargs)
    if not res:
        return None
    async with aiofiles.open(
        file, "a"
    ) as f:  # NOTE write out using "append" mode
        for k, v in res.items():
            await f.write(",".join([k, v]) + "\n")  # for further parsing
        #  logger.info("Wrote results for chaps: %s", k)


async def bulk_crawl_and_write(file: IO, urls: str, **kwargs) -> None:
    """Crawl and write concurrently to `file` for multiple `urls`."""
    async with ClientSession() as session:
        tasks = []
        for url in urls:
            tasks.append(
                write_one(file=file, url=url, session=session, **kwargs)
            )
        await asyncio.gather(*(tasks))


if __name__ == "__main__":

    #  # 三部曲, 每部共有 54 章,
    #  #  1, 1_2，1_3 分别为第一章的第一、第二、第三小节; 下同
    #  first_part = base_url + "diyibu/"   # 共有 54 章, 序号 1 -> 54
    #  second_part = base_url + "dierbu/"  # 共有 54 章, 序号 55 -> 108
    #  third_part = base_url + "disanbu/"  # 共有 54 章, 序号 109 -> 162
    #
    #  def demo1():
    #      part1_chap1 = requests.get(first_part + "1.html", headers=headers)
    #      # Create a BeautifulSoup object
    #      soup = BeautifulSoup(part1_chap1.text, "html.parser")
    #      print(soup.select_one(".span12 p:nth-of-type(1)").text)
    #
    #  def demo2():
    #      part1_chap1_2 = requests.get(first_part + "1_2.html", headers=headers)
    #      # Create a BeautifulSoup object
    #      soup = BeautifulSoup(part1_chap1_2.text, "html.parser")
    #      print(soup.select_one(".span12").text)  # works; 2024-04-07 Sun
    #
    #  def demo3():
    #      part1_chap1 = requests.get(first_part + "1.html", headers=headers)
    #      # Create a BeautifulSoup object
    #      soup = BeautifulSoup(part1_chap1.text, "html.parser")
    #      print(soup.select_one(".pagination b:nth-of-type(2)").text)
    #
    #  #  demo1()
    #  #  demo2()
    #  #  demo3()

    def get_part_num(x):
        # compare to chapter number (float)
        if x < 55:  # part one of the book
            return 1
        elif x < 109:  # part two
            return 2
        elif x < 163:  # part three
            return 3

    assert sys.version_info >= (3, 7), "Script requires Python 3.7+"

    try:
        urls_d = joblib.load("data/pfdsj_urls_dict.pkl")  # ["diyibu"]
    except FileNotFoundError:
        urls_d = produce_urls()  # ["diyibu"]

    urls = []
    for k, v in urls_d.items():
        urls.extend(v)

    # 乱序写入文本，CSV格式，可以进一步用pandas进行清洗
    outpath = "data/PingFanDeShiJie_3in1_out_of_order.csv"
    asyncio.run(bulk_crawl_and_write(file=outpath, urls=urls))

    # 整理文本内容顺序：
    columns = ["chapter", "text"]
    data = pd.read_csv(outpath)
    data.columns = columns
    data["chap_num"] = (
        data["chapter"].apply(lambda x: ".".join(x.split("_"))).astype(float)
    )
    data["part"] = data["chap_num"].apply(get_part_num)
    result = data.sort_values("chap_num")
    result.to_csv("data/PingFanDeShiJie_3in1_ordered.csv")
