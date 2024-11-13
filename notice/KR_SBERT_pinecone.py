from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from google.oauth2 import service_account
from googleapiclient.discovery import build
import torch
import logging
import shutil
import os
from pathlib import Path
from tqdm import tqdm
from KR_SBERT_fine_tuning import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SERVICE_ACCOUNT_FILE = 'swengineer-e9e6a19f0a3d.json'
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
MODEL_NAME = 'snunlp/KR-SBERT-V40K-klueNLI-augSTS'
PINECONE_API_KEY = "1734fc56-9964-4232-a412-50e211980310"
PINECONE_INDEX_NAME = "skku-notice"

class NoticeVectorDB:
    def __init__(self, api_key: str, index_name: str):
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        
        # GPU 사용 가능시 GPU 사용
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # SBERT 모델 로드
        try:
            self.model = SentenceTransformer(MODEL_NAME)
            self.model.to(self.device)
            logger.info(f"Model loaded: {MODEL_NAME}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def read_spreadsheet(self, spreadsheet_id: str, range_name: str) -> pd.DataFrame:
        try:
            creds = service_account.Credentials.from_service_account_file(
                SERVICE_ACCOUNT_FILE, scopes=SCOPES)
            
            service = build('sheets', 'v4', credentials=creds)
            sheet = service.spreadsheets()
            result = sheet.values().get(spreadsheetId=spreadsheet_id,
                                      range=range_name).execute()
            values = result.get('values', [])
            
            df = pd.DataFrame(values, columns=['name', 'ArticleNo', 'category', 'title', 
                                             'notice_date', 'url', 'content'])
            logger.info(f"Loaded {len(df)} rows from spreadsheet")
            return df
            
        except Exception as e:
            logger.error(f"Error reading spreadsheet: {e}")
            raise

    def preprocess_text(self, row: pd.Series) -> str:
        # 제목 가중치
        return f"{row['title']} {row['title']} {row['name']} {row['category']} {row['content']}"

    def get_embeddings(self, texts: list) -> torch.Tensor:
        try:
            with torch.no_grad():
                embeddings = self.model.encode(texts, convert_to_tensor=True)
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

    def upload_to_pinecone(self, df: pd.DataFrame, batch_size: int = 32):
        try:
            index = self.pc.Index(self.index_name)
            total_batches = (len(df) + batch_size - 1) // batch_size
            
            for i in tqdm(range(0, len(df), batch_size), desc="Uploading to Pinecone"):
                batch_df = df.iloc[i:i+batch_size]
                
                texts = [self.preprocess_text(row) for _, row in batch_df.iterrows()]
                embeddings = self.get_embeddings(texts).cpu().numpy()
                
                vectors = [
                    {
                        "id": str(row['ArticleNo']),
                        "values": embedding.tolist(),
                        "metadata": {
                            "name": row['name'],
                            "category": row['category'],
                            "title": row['title'],
                            "notice_date": row['notice_date'],
                            "url": row['url'],
                            "content": row['content']
                        }
                    }
                    for (_, row), embedding in zip(batch_df.iterrows(), embeddings)
                ]
                
                index.upsert(vectors=vectors)
                
            logger.info(f"Successfully uploaded {len(df)} documents to Pinecone")
            
        except Exception as e:
            logger.error(f"Error uploading to Pinecone: {e}")
            raise

    def find_similar_notices(self, query: str, top_k: int = 5):
        try:
            # 쿼리 임베딩 생성
            query_embedding = self.model.encode(query).tolist()
            
            # Pinecone 검색
            index = self.pc.Index(self.index_name)
            results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching similar notices: {e}")
            raise
        
    def fine_tune(self, df: pd.DataFrame, epochs: int = 10, output_path: str = 'finetuned-kr-sbert-notice'):
        try:
            # 기존 모델 있으면 dir 삭제
            if os.path.exists(output_path):
                logger.info(f"Removing existing model directory: {output_path}")
                shutil.rmtree(output_path)
            
            # output dir 생성
            Path(output_path).mkdir(parents=True, exist_ok=True)
            logger.info(f"Created new model directory: {output_path}")
            
            logger.info("Preparing training examples...")
            training_generator = NoticeTrainingDataGenerator()
            training_examples = training_generator.create_combined_training_examples(df)
            
            logger.info(f"Created {len(training_examples)} training examples")
            
            logger.info("Setting up training...")
            train_dataloader = DataLoader(
                training_examples,
                shuffle=True,
                batch_size=16
            )       
            train_loss = losses.CosineSimilarityLoss(self.model)
            warmup_steps = int(len(train_dataloader) * 0.1)
            
            training_args = {
                'train_objectives': [(train_dataloader, train_loss)],
                'epochs': epochs,
                'warmup_steps': warmup_steps,
                'output_path': output_path,
                'use_amp': False,
                'optimizer_class': torch.optim.AdamW,
                'optimizer_params': {'lr': 2e-5},
                'weight_decay': 0.01,
                'scheduler': 'WarmupLinear',
                'show_progress_bar': True
            }
            
            logger.info("Starting fine-tuning...")
            self.model.fit(**training_args)
            
            logger.info(f"Fine-tuning completed. Model saved to {output_path}")
            self.model = SentenceTransformer(output_path)
            self.model.to(self.device)
            logger.info("Fine-tuned model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error during fine-tuning: {e}")
            raise

def main():
    try:
        # vector db 초기화
        logger.info("Initializing NoticeVectorDB...")
        vdb = NoticeVectorDB(
            api_key=PINECONE_API_KEY,
            index_name=PINECONE_INDEX_NAME,
        )
        
        # 데이터 로드
        logger.info("Loading spreadsheet data...")
        spreadsheet_id = "1BrbQzpoxxhxBTQcyRCPIrZ32mgY_fNxAmVmCxpV7Rw0"
        df = vdb.read_spreadsheet(spreadsheet_id, "sheet1!A:G")
        
        # Fine-tuning 실행
        logger.info("Starting fine-tuning process...")
        vdb.fine_tune(df, epochs=10)
        
        # Pinecone에 업로드
        logger.info("Starting upload to Pinecone...")
        vdb.upload_to_pinecone(df)
        
        # 테스트 검색
        logger.info("Testing search functionality...")
        test_queries = [
            "소프트웨어학과 졸업요건을 알려줘",
            "장학금 신청 방법 알려줘",
            "기숙사 신청 언제해?"
        ]

        for query in test_queries:
            results = vdb.find_similar_notices(query)
            print(f"\nResults for query: {query}")
            for match in results["matches"][:3]:
                print(f"부서: {match['metadata']['name']}")
                print(f"제목: {match['metadata']['title']}")
                print(f"유사도: {match['score']:.4f}")
                print("-" * 50)
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    main()