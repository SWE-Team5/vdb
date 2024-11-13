from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import logging
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FINETUNED_MODEL_PATH = 'finetuned-kr-sbert-notice'  # fine-tuning model path
PINECONE_API_KEY = "1734fc56-9964-4232-a412-50e211980310"
PINECONE_INDEX_NAME = "skku-notice"

class NoticeSearcher:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Fine-tuned model load
        try:
            self.model = SentenceTransformer(FINETUNED_MODEL_PATH)
            self.model.to(self.device)
            logger.info("Fine-tuned model loaded successfully")
        except Exception as e:
            logger.warning(f"Fine-tuned model not found: {e}")
            logger.info("Loading base model instead...")
            self.model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')
            self.model.to(self.device)
        
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index = self.pc.Index(PINECONE_INDEX_NAME)

    def find_similar_notices(self, query: str, top_k: int = 5):
        try:
            logger.info(f"Processing query: {query}")
            
            query_embedding = self.model.encode(query).tolist()
            
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            print(f"\n검색 결과 - 질문: {query}\n")

            
            return results
            
        except Exception as e:
            logger.error(f"Error during search: {e}")
            raise

def pinecone_main():
    try:
        searcher = NoticeSearcher()
        
        # test query
        test_queries = "소프트웨어학과 졸업 관련 공지 있어?"
            # "현재 신청할 수 있는 장학금이 있을까요?"
            # "기숙사 관련 공지가 있어?"
            # "근로장학 공지사항을 알려줘"
        
        ret = searcher.find_similar_notices(test_queries)
        
        for match in ret["matches"]:
            print(f"제목: {match['metadata']['title']}")
            print(f"부서: {match['metadata']['name']}")
            print(f"날짜: {match['metadata']['notice_date']}")
            print(f"URL: {match['metadata']['url']}")
            print(f"유사도: {match['score']:.4f}")
            print("-" * 50)
            
        print("\n" + "="*70 + "\n")
        return ret
            
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    pinecone_main()