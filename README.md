# SKKU Notice Vector Database Project

## Overview
이 프로젝트는 성균관대학교 공지사항을 벡터 데이터베이스화하여 효율적인 검색과 추천을 제공하는 시스템입니다. KR-SBERT를 사용하여 한국어 공지사항을 임베딩하고, Pinecone을 통해 벡터 검색을 구현했습니다.

## Project Structure
```
notice/
├── KR_SBERT_fine_tuning.py    # Fine-tuning 관련 클래스와 함수
├── KR_SBERT_pinecone.py       # 메인 벡터 데이터베이스 구현
├── test_query.py              # 검색 테스트 스크립트
├── create_index.py            # Pinecone 인덱스 생성
├── get_notice.py              # 공지사항 google spreadsheet에 추가
├── vdb.py                     # pincone upload with finetuned model
└── swengineer-e9e6a19f0a3d.json  # Google Sheets API 인증 파일
```

## Requirements

### System Requirements
- Python 3.10
- CUDA 12.3 (GPU 사용 시, 로컬(RTX 3060 기준))
- WSL2 (Windows 환경에서 실행 시)

### Dependencies
```
torch==2.0.1
transformers==4.34.0
sentence-transformers==2.2.2
pinecone-client
google-api-python-client
pandas
tqdm
```

## Environment Setup

### 1. Conda Environment 설정
```bash
# Conda 설치
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# 환경 생성
conda create -n notice python=3.10
conda activate notice

# PyTorch 및 CUDA 설치 
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# 필요한 패키지 설치
conda install -c conda-forge transformers
conda install -c conda-forge sentence-transformers
pip install pinecone-client 
pip install google-api-python-client 
pip install pandas 
pip install tqdm
```

### 2. Google Sheets API 설정
- Google Cloud Console에서 서비스 계정 생성
- 생성된 JSON 키 파일을 'swengineer-e9e6a19f0a3d.json'으로 저장
- Google Sheets API 활성화

### 3. Pinecone 설정
- Pinecone 계정 생성
- API 키 발급
- 768차원 코사인 유사도 기반 인덱스 생성

## Usage

### 1. Fine-tuning
```bash
python KR_SBERT_pinecone.py
```
- 공지사항 데이터로 KR-SBERT 모델을 fine-tuning
- 학습된 모델은 'finetuned-kr-sbert-notice' 디렉토리에 저장

### 2. 검색 테스트
```bash
python test_query.py
```
- Fine-tuning된 모델을 사용하여 공지사항 검색
- 질문에 대한 상위 5개의 관련 공지사항 반환

## Model Details
- Base Model: snunlp/KR-SBERT-V40K-klueNLI-augSTS
- Embedding Dimension: 768
- Fine-tuning Strategy:
  - category 기반 유사도 학습
  - query-name(부서) 쌍 학습
  - title-content 관계 학습

## Notes
- GPU 사용 시 CUDA 설정 필요
- WSL 환경에서 실행 시 NVIDIA 드라이버 설정 필요(e.g. nvidia-smi)

## License
This project is licensed under the MIT License

## Contact
For any questions or issues, please contact [hoeo456@g.skku.edu]