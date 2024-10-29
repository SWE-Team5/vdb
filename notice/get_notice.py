import schedule
from datetime import datetime
from crawling import *
import re

def fetch_notice_data(urls):
    sheet = client.open_by_key(SPREADSHEET_ID).worksheet(SHEET_NAME)

    now = datetime.now()
    print("fetching notice data...")
    print(f"현재 시각: {now}")
    
    for main_url, name in urls:
        try:
            response = requests.get(main_url, headers=headers, timeout=10)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching {main_url}: {e}")
            continue
        
        if name == "dorm":
            fetch_dorm_notice_data(main_url, name, response, sheet)
            continue
        
        soup = BeautifulSoup(response.text, "html.parser")
        notices = soup.find_all('li', class_='')
        
        for notice in notices:
            category_tag = notice.find('span', class_='c-board-list-category')
            if not category_tag:
                continue
            category = category_tag.get_text(strip=True)
            title_tag = notice.find('a')
            url_tag = title_tag.get('href')
            url = main_url + url_tag 
            if not title_tag:
                continue
            title = title_tag.get_text(strip=True)
            info_items = notice.find('ul').find_all('li')
            if len(info_items) >= 4:
                tmp_no = info_items[0].get_text(strip=True)
                if tmp_no.startswith("No."):
                    no = tmp_no.replace("No.", "").strip()
                elif tmp_no == "공지" or tmp_no == '':
                    no = -1
                else:
                    no = tmp_no
                    
                match = re.search(r'articleNo=(\d+)', url_tag)
                if match:
                    ArticleNo = match.group(1)
                else:
                    ArticleNo = -1
                
                date = info_items[2].get_text(strip=True)
                notice_date = datetime.strptime(date, '%Y-%m-%d')
                content = get_notice_details(url)
                ArticleNoCell = sheet.find(str(ArticleNo))
                
                if ArticleNoCell and no != -1:
                    print(f"{name} 공지사항 {ArticleNo}가 이미 스프레드시트에 존재합니다.")
                    continue
                elif ArticleNoCell and no == -1:
                    continue
                    
                append_row_with_retry(sheet, [
                    name,
                    ArticleNo,
                    category,
                    title,
                    notice_date.strftime('%Y-%m-%d'),
                    url,
                    content
                ])
                print(f"{name} 공지사항 {ArticleNo}가 스프레드시트에 추가되었습니다.")
        
        time.sleep(300)
    
    print("모든 공지사항 업데이트 완료.")
    
def fetch_dorm_notice_data(main_url, name, response, sheet):
    
    soup = BeautifulSoup(response.text, 'html.parser')
    notice_table = soup.find('table', class_='list_table')
    notices = notice_table.find_all('tr') if notice_table else []
    
    for notice in notices:
        columns = notice.find_all('td')
        
        if len(columns) < 5:
            continue
        
        no = columns[0].get_text(strip=True)
        if no == '':
            no = -1
        category = columns[1].get_text(strip=True)
        title = columns[2].find('a').get_text(strip=True)
        url_tag = columns[2].find('a')['href']
        url = main_url + url_tag 
        date = columns[4].get_text(strip=True)
        notice_date = datetime.strptime(date, '%Y-%m-%d')

        match = re.search(r'article_no=(\d+)', url_tag)
        if match:
            ArticleNo = match.group(1)
        else:
            ArticleNo = -1
        
        content = get_dorm_notice_details(url)
        if content == '':
            content = title

        ArticleNoCell = sheet.find(str(ArticleNo))
        
        if ArticleNoCell and no != -1:
            print(f"{name} 공지사항 {ArticleNo}가 이미 스프레드시트에 존재합니다.")
            continue
        elif ArticleNoCell and no == -1:
            continue
            
        append_row_with_retry(sheet, [
            name,
            ArticleNo,
            category,
            title,
            notice_date.strftime('%Y-%m-%d'),
            url,
            content
        ])
        print(f"{name} 공지사항 {ArticleNo}가 스프레드시트에 추가되었습니다.")

schedule.every(60).minutes.do(fetch_notice_data, urls)

fetch_notice_data(urls)

while True:
    schedule.run_pending()
    time.sleep(1)
