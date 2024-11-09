import schedule
from datetime import datetime
from crawling import *
import re
import pinecone
from sentence_transformers import SentenceTransformer

# Pinecone 관련 설정
PINECONE_API_KEY = "your_pinecone_api_key"
PINECONE_INDEX_NAME = "skku_notice"

# 문서 임베딩 모델 로드
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

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
            # ... (기존 코드 유지)
            
            # Google Spreadsheet에 데이터 추가
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
            
            # Pinecone에 데이터 추가
            add_to_pinecone(name, ArticleNo, title, content)
        
        time.sleep(300)
    
    print("모든 공지사항 업데이트 완료.")

def add_to_pinecone(name, article_no, title, content):
    """Pinecone 벡터 DB에 데이터 추가"""
    try:
        # Pinecone 클라이언트 생성
        pinecone.init(api_key=PINECONE_API_KEY)
        index = pinecone.Index(PINECONE_INDEX_NAME)
        
        # 텍스트 임베딩
        text = f"{name} {title} {content}"
        embedding = model.encode(text).tolist()
        
        # Pinecone에 데이터 업로드
        index.upsert(vectors=[{
            "id": str(article_no),
            "values": embedding,
            "metadata": {
                "name": name,
                "article_no": article_no,
                "title": title,
                "content": content
            }
        }])
        
        print(f"Pinecone에 {article_no} 추가 완료")
    except Exception as e:
        print(f"Pinecone 업데이트 중 오류 발생: {e}")

schedule.every(60).minutes.do(fetch_notice_data, urls)

fetch_notice_data(urls)

while True:
    schedule.run_pending()
    time.sleep(1)