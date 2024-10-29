import torch
from transformers import AutoTokenizer, AutoModel
from pinecone import Pinecone

PINECONE_API_KEY = "1734fc56-9964-4232-a412-50e211980310"
PINECONE_INDEX_NAME = "skku-notice"

def find_similar_notices(query):
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(PINECONE_INDEX_NAME)
        
        # vector embedding
        tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
        model = AutoModel.from_pretrained("klue/bert-base")
        inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True)
        
        with torch.no_grad():
            outputs = model(**inputs)
            query_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy().astype('float32').tolist()

        results = index.query(queries=[query_embedding], top_k=5, include_metadata=True)
        
        for result in results:
            metadata = result.metadata
            print(f"제목: {metadata['title']}")
            print(f"날짜: {metadata['notice_date']}")
            print(f"URL: {metadata['url']}")
            print(f"유사도: {result.score}")
            print("---")
        
        print(f"질문: {query}")
    except Exception as e:
        print(f"Pinecone 검색 중 오류 발생: {e}")

if __name__ == "__main__":
    query = "학사 취업"
    find_similar_notices(query)