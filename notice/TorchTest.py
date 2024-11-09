import torch
print("CUDA available:", torch.cuda.is_available())
print("CUDA device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

from sentence_transformers import SentenceTransformer
print("Testing Korean SBERT loading...")
model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')
print("Model loaded successfully!")

# 간단한 테스트
test_sentences = ["안녕하세요", "반갑습니다"]
embeddings = model.encode(test_sentences)
print("Embeddings shape:", embeddings.shape)