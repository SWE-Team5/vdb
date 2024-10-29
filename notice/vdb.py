from pinecone import Pinecone
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from googleapiclient.discovery import build
from google.oauth2 import service_account

SERVICE_ACCOUNT_FILE = 'swengineer-e9e6a19f0a3d.json'
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
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
    tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
    model = AutoModel.from_pretrained("klue/bert-base")
    embeddings = []
    
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy().astype('float32')
            embeddings.append(cls_embedding)
    
    return embeddings

def upload_to_pinecone(df):
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)
    
    batch_size = 100
    for i in range(0, len(df), batch_size):
        batch_df = df.iloc[i:i+batch_size]
        
        texts = [f"{row['name']} {row['content']}" for _, row in batch_df.iterrows()]
        embeddings = get_embeddings(texts)
        
        vectors = []
        for j, (_, row) in enumerate(batch_df.iterrows()):
            vectors.append({
                "id": str(row['ArticleNo']),
                "values": embeddings[j].tolist(),
                "metadata": {
                    "name": row['name'],
                    "category": row['category'],
                    "title": row['title'],
                    "notice_date": row['notice_date'],
                    "url": row['url'],
                    "content": row['content']
                }
            })
        
        index.upsert(vectors=vectors)
        
if __name__ == "__main__":
    spreadsheet_id = "1BrbQzpoxxhxBTQcyRCPIrZ32mgY_fNxAmVmCxpV7Rw0"
    range_name = "sheet1!A:G"
    
    df = read_spreadsheet(spreadsheet_id, range_name)
    
    print("---")
    print(df.columns)
    print("---")
    print(df.head())
    print("---")
    print(df.info())
    print("---")
    
    print("data loaded")
    upload_to_pinecone(df)
    print("data uploaded")