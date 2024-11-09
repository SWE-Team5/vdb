import requests
from bs4 import BeautifulSoup
from google.oauth2.service_account import Credentials
import gspread
import time

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"
}

skku_url = "https://www.skku.edu/skku/campus/skk_comm/notice01.do"
cse_url = "https://cse.skku.edu/cse/notice.do"
py_url = "https://physics.skku.ac.kr/physics/notice/notice.do"
ai_url = "https://xai.skku.edu/skkuaai/notice.do"
biz_url = "https://biz.skku.edu/bizskk/notice.do"
dorm_url = "https://dorm.skku.edu/dorm_suwon/notice/notice_all.jsp"

using_urls = [skku_url, cse_url, py_url, ai_url, biz_url, dorm_url]
notice_name = ["skku", "cse", "physics", "ai", "biz", "dorm"]

urls = list(zip(using_urls, notice_name))

# Google Sheets API 인증 설정
SERVICE_ACCOUNT_FILE = 'swengineer-e9e6a19f0a3d.json'
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']

creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
client = gspread.authorize(creds)

# 스프레드시트 및 시트 설정
SPREADSHEET_ID = '1BrbQzpoxxhxBTQcyRCPIrZ32mgY_fNxAmVmCxpV7Rw0'
SHEET_NAME = 'sheet1'
# SHEET_NAME = 'test'

MAX_RETRIES = 5  # 재시도 횟수 제한

def get_notice_details(notice_url):

    try:
        response = requests.get(notice_url, headers=headers)
        response.raise_for_status()
        detail_soup = BeautifulSoup(response.text, "html.parser")
        
        content = detail_soup.find('div', class_='content')
        if content:
            return content.get_text(strip=True)
        else:
            return "Not found"
    except Exception as ex:
        print("content err")
        print(f"{ex}")
        return "Error"
    
def get_dorm_notice_details(notice_url):
    try:
        response = requests.get(notice_url, headers=headers)
        response.raise_for_status()
        detail_soup = BeautifulSoup(response.text, "html.parser")
        
        content = detail_soup.find('div', id='article_text')
        if content:
            return content.get_text(strip=True)
        else:
            return "Not found"
    except Exception as ex:
        print(f"dorm content err: {ex}")
        return "Error"
    
def append_row_with_retry(sheet, row_data):
    retries = 0
    while retries < MAX_RETRIES:
        try:
            sheet.append_row(row_data)
            break
        except gspread.exceptions.APIError as e:
            retries += 1
            print(f"API 오류 발생: {e}, 재시도 {retries}/{MAX_RETRIES}")
            time.sleep(5)
        except Exception as e:
            print(f"다른 오류 발생: {e}")
            break