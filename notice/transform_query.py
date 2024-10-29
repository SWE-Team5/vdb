from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

PINECONE_API_KEY = "1734fc56-9964-4232-a412-50e211980310"
PINECONE_INDEX_NAME = "skku-notice"

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def find_similar_notices(query):
    try:
        # Pinecone 클라 초기화
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(PINECONE_INDEX_NAME)
        
        # vector embedding
        query_embedding = model.encode(query).tolist()
        print(f"Query embedding: {query_embedding}")

        results = index.query(vector=query_embedding, top_k=5, include_metadata=True)
        
        for match in results["matches"]:
            print(f"제목: {match['metadata']['title']}")
            print(f"날짜: {match['metadata']['notice_date']}")
            print(f"URL: {match['metadata']['url']}")
            print(f"유사도: {match['score']}")
            print("---")
            
        print(f"질문 : {query}")
    except Exception as e:
        print(f"Pinecone 검색 중 오류 발생: {e}")

# test
test_query = "학사 취업"
find_similar_notices(test_query)