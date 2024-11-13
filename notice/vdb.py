from pinecone import Pinecone
import pandas as pd
from sentence_transformers import SentenceTransformer
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from google.oauth2 import service_account

SERVICE_ACCOUNT_FILE = 'swengineer-e9e6a19f0a3d.json'
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
FINETUNED_MODEL_PATH = 'finetuned-kr-sbert-notice'  # fine-tuning model path
PINECONE_API_KEY = "1734fc56-9964-4232-a412-50e211980310"
PINECONE_INDEX_NAME = "skku-notice"

# Google Spreadsheet 데이터 읽기
def read_spreadsheet(spreadsheet_id, range_name):
    creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    
    service = build('sheets', 'v4', credentials=creds)
    sheet = service.spreadsheets()
    result = sheet.values().get(spreadsheetId=spreadsheet_id,
                              range=range_name).execute()
    values = result.get('values', [])
    
    df = pd.DataFrame(values, columns=['name', 'ArticleNo', 'category', 'title', 'notice_date', 'url', 'content'])
    
    return df

# 텍스트 임베딩 함수
def get_embeddings(texts):
    model = SentenceTransformer(FINETUNED_MODEL_PATH)
    embeddings = model.encode(texts)
    return embeddings

def upload_to_pinecone(df, index_name):
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(index_name)
    
    batch_size = 100
    for i in range(0, len(df), batch_size):
        batch_df = df.iloc[i:i+batch_size]
        
        texts = [f"{row['name']} {row['content']}" for _, row in batch_df.iterrows()]
        embeddings = get_embeddings(texts)
        
        vectors = []
        for j, (_, row) in enumerate(batch_df.iterrows()):
            vectors.append({
                "id": str(row['ArticleNo']),  # 고유 ID
                "values": embeddings[j].tolist(),  # 벡터 값
                "metadata": { 
                    "name": row['name'],
                    "category": row['category'],
                    "title": row['title'],
                    "notice_date": row['notice_date'],
                    "url": row['url'],
                    "content": row['content']
                }
            })
        
        # Pinecone에 업로드
        index.upsert(vectors=vectors)
        
if __name__ == "__main__":
    spreadsheet_id = "1BrbQzpoxxhxBTQcyRCPIrZ32mgY_fNxAmVmCxpV7Rw0"
    range_name = "sheet1!A:G"
    
    df = read_spreadsheet(spreadsheet_id, range_name)
    
    print(df.columns)
    print(df.head())
    print(df.info())
    
    print("data loaded")
    upload_to_pinecone(df, PINECONE_INDEX_NAME)
    print("data uploaded")