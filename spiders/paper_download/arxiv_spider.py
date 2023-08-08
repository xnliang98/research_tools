import re
import os
import io
import sys
import time
import requests
from bs4 import BeautifulSoup
import json
from pdf2image import convert_from_bytes
from PIL import Image
from datetime import datetime
from tqdm import tqdm

def remove_symbols(title):
    new_title = re.sub('[^a-zA-Z ]', ' ', title)
    return new_title

def get_formatted_date():
    # 获取当前日期
    current_date = datetime.now()
    # 将日期格式化成 YYYY-MM-DD 格式的字符串
    formatted_date = current_date.strftime("%Y-%m-%d")
    return formatted_date


def download_pdf_image(url, title):
    # 保存第一页图片
    date_str = get_formatted_date()
    title = remove_symbols(title)
    os.makedirs(f'./{date_str}/', exist_ok=True)
    path = f'{date_str}/{date_str}_{title}.png'

    if os.path.exists(path):
        return

    # 下载PDF文件
    response = requests.get(url)

    # 将PDF文件转换为图片
    images = convert_from_bytes(response.content)

    # 获取第一页图片
    first_page = images[0]
    first_page.save(path, 'PNG')

def download_file(url, title, saved_path):
    send_headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36",
        "Connection": "keep-alive",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.8"}
    date_str = get_formatted_date()
    # title = remove_symbols(title)
    os.makedirs(f'{saved_path}/{date_str}/pdf_files', exist_ok=True)
    os.makedirs(f'{saved_path}/{date_str}/img_files', exist_ok=True)
    path = f'{saved_path}/{date_str}/pdf_files/{title}.pdf'
    img_path = f'{saved_path}/{date_str}/img_files/{title}.png'

    req = requests.get(url, headers=send_headers)  # 通过访问互联网得到文件内容
    bytes_io = io.BytesIO(req.content)  # 转换为字节流
    with open(path, 'wb') as file:
        file.write(bytes_io.getvalue())  # 保存到本地

    # 将PDF文件转换为图片
    images = convert_from_bytes(req.content)
    # 获取第一页图片
    first_page = images[0]
    first_page.save(img_path, 'PNG')
    # time.sleep(1) # 最好做一个休眠
    return bytes_io

def crawl_html(url):
    response = requests.get(url)
    html = response.text

    soup = BeautifulSoup(html, 'html.parser')
    dl_tags = soup.find_all('dl')

    if len(dl_tags) > 0:
        dl_tag = dl_tags[0]
        dt_tags = dl_tag.find_all('dt')
        dd_tags = dl_tag.find_all('dd')
        
        result = []
        for dt_tag, dd_tag in zip(dt_tags, dd_tags):
            a_tag = dt_tag.find('a', {'title': 'Abstract'})
          
            if a_tag:
                arxiv_id = a_tag['href'].split('/')[-1]
                title_element = dd_tag.find('div', class_='list-title mathjax')
                title = title_element.text.strip().replace('Title:', ' ').strip()
                title = " ".join(title.split())
                authors_element = dd_tag.find('div', class_='list-authors')
                authors = authors_element.text.strip().replace('Authors:', ' ').replace("\n", "").strip()
             
                comments_element = dd_tag.find('div', class_='list-comments mathjax')
                if comments_element:
                    comments = comments_element.text.strip().replace('Comments:', ' ').strip()
                else:
                    comments = ""
                # https://arxiv.org /pdf/2308.02482  .pdf
                # /pdf/2308.02482
                # pdf_link = dt_tag.find('a', {'title': 'Download PDF'})['href']
                pdf_link = a_tag['href'].replace("abs", "pdf")
                pdf_link = f"https://arxiv.org{pdf_link}.pdf"

                result.append({'arxiv_id': arxiv_id, 'title': title, 'pdf_link': pdf_link, 'comments': comments, "authors": authors})
        return result
    else:
        return None

url = 'https://arxiv.org/list/cs.CL/pastweek?show=500'  # 替换为你要爬取的url
result = crawl_html(url)

date_str = get_formatted_date()
# saved_path = "/arxiv_daily"
saved_path = sys.argv[1]
os.makedirs(f'{saved_path}/{date_str}/', exist_ok=True)


# 保存结果到文件
json_path = f'{saved_path}/{date_str}/arxiv_{date_str}.json'
with open(json_path, 'w', encoding='utf-8') as file:
    json.dump(result, file, ensure_ascii=False, indent=2)

total = len(result)

md_path = f'{saved_path}/{date_str}/arxiv_{date_str}.md'
with open(md_path, 'w', encoding='utf-8') as fw:
    fw.write(f"# {date_str} Total {total} papers\n\n")
    for res in result:
        title = res['title']
        title = remove_symbols(title)
        title = " ".join(title.split())
        title = title.replace(" ", "_")

        md_block = f"## {res['title']}\n- arXiv id: {res['arxiv_id']}\n- PDF LINK: {res['pdf_link']}\
                \n- authors: {res['authors']}\n- comments: {res['comments']}\n- [PDF FILE](./pdf_files/{title}.pdf)\n\n ![fisrt page](./img_files/{title}.png)\n\n\n"
        fw.writelines(md_block)

err_file = open(f'{saved_path}/{date_str}/err.log', "a")
for i, js in enumerate(result):
    title = js['title']
    title = remove_symbols(title)
    title = " ".join(title.split())
    title = title.replace(" ", "_")

    arxiv_id = js['arxiv_id']
    pdf_link = js['pdf_link']
    print(f"processing {i+1}/{total} {title} ...")
    if os.path.exists(f'{saved_path}/{date_str}/pdf_files/{title}.pdf'):
        continue
    # download_pdf_image(pdf_link, title)
    try:
        download_file(pdf_link, title, saved_path)
    except Exception:
        err_file.write(f"下载出错： {i+1}/{total} {title} ...")
        continue

print('\n\ndone!!!')