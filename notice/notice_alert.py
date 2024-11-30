import schedule
from datetime import datetime
from crawling import *
import re
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# VDB 관련 상수 추가
FINETUNED_MODEL_PATH = 'finetuned-kr-sbert-notice'
PINECONE_API_KEY = "1734fc56-9964-4232-a412-50e211980310"
PINECONE_INDEX_NAME = "skku-notice"

# 이메일 설정
MAIL_SERVER = "smtp.gmail.com"
MAIL_PORT = 587
SENDER_EMAIL = "hoeo159@gmail.com"
APP_PASSWORD = "miminho?159"
RECIPIENT_EMAIL = "your_email@example.com"  # 실제 사용할 이메일 주소로 변경하세요

# 키워드 설정
KEYWORDS = ["소프트웨어학과"]
SIMILARITY_THRESHOLD = 0.45

# 모델 로드
model = SentenceTransformer(FINETUNED_MODEL_PATH)

# Pinecone 초기화
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

def get_embedding(text):
    embedding = model.encode(text)
    return embedding

def check_keyword_similarity(notice_text, keywords):
    notice_embedding = get_embedding(notice_text)
    keyword_embeddings = model.encode(keywords)
    
    matched_keywords = []
    for i, keyword_embedding in enumerate(keyword_embeddings):
        similarity = float(notice_embedding @ keyword_embedding)
        if similarity >= SIMILARITY_THRESHOLD:
            matched_keywords.append((keywords[i], similarity))
    
    return matched_keywords

def send_email_alert(notice_data, matched_keywords):
    try:
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = RECIPIENT_EMAIL
        msg['Subject'] = f"[공지사항 알림] {notice_data['title']}"
        
        keyword_text = ', '.join([f"{kw}({score:.2f})" for kw, score in matched_keywords])
        
        body = f"""
        안녕하세요.
        설정하신 키워드와 관련된 새로운 공지사항이 등록되었습니다.
        
        [공지사항 정보]
        제목: {notice_data['title']}
        등록일: {notice_data['notice_date']}
        카테고리: {notice_data['category']}
        게시부서: {notice_data['name']}
        
        관련 키워드 (유사도): {keyword_text}
        
        원문 링크: {notice_data['url']}
        
        [공지사항 내용]
        {notice_data['content'][:300]}...
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        with smtplib.SMTP(MAIL_SERVER, MAIL_PORT) as server:
            server.starttls()
            server.login(SENDER_EMAIL, APP_PASSWORD)
            server.send_message(msg)
            
        print(f"Alert email sent for notice: {notice_data['title']}")
    except Exception as e:
        print(f"Error sending email: {e}")

def upload_to_pinecone(row_data):
    text = f"{row_data['name']} {row_data['content']}"
    embedding = get_embedding(text)
    
    vector = {
        "id": str(row_data['ArticleNo']),
        "values": embedding.tolist(),
        "metadata": {
            "name": row_data['name'],
            "category": row_data['category'],
            "title": row_data['title'],
            "notice_date": row_data['notice_date'],
            "url": row_data['url'],
            "content": row_data['content']
        }
    }
    
    # Pinecone에 업로드
    index.upsert(vectors=[vector])
    print(f"Vector uploaded to Pinecone for article {row_data['ArticleNo']}")
    
    # 키워드 매칭 확인 및 이메일 발송
    notice_text = f"{row_data['title']} {row_data['content']}"
    matched_keywords = check_keyword_similarity(notice_text, KEYWORDS)
    
    if matched_keywords:
        send_email_alert(row_data, matched_keywords)

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
                    
                # 스프레드시트에 추가
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
                
                # Pinecone에 추가 및 알림 발송
                row_data = {
                    'name': name,
                    'ArticleNo': ArticleNo,
                    'category': category,
                    'title': title,
                    'notice_date': notice_date.strftime('%Y-%m-%d'),
                    'url': url,
                    'content': content
                }
                upload_to_pinecone(row_data)
        
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
            
        # 스프레드시트에 추가
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
        
        # Pinecone에 추가 및 알림 발송
        row_data = {
            'name': name,
            'ArticleNo': ArticleNo,
            'category': category,
            'title': title,
            'notice_date': notice_date.strftime('%Y-%m-%d'),
            'url': url,
            'content': content
        }
        upload_to_pinecone(row_data)

if __name__ == "__main__":
    # 이메일 주소 입력받기
    RECIPIENT_EMAIL = input("알림을 받을 이메일 주소 : ")
    
    schedule.every(60).minutes.do(fetch_notice_data, urls)
    fetch_notice_data(urls)

    while True:
        schedule.run_pending()
        time.sleep(1)