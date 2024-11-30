from sentence_transformers import InputExample, losses, InputExample
from torch.utils.data import DataLoader
from typing import List, Dict, Tuple
import pandas as pd
import random

notice_name = ["skku", "cse", "physics", "ai", "biz", "dorm"]

class NoticeTrainingDataGenerator:
    def __init__(self):
        # query-keyword mapping
        self.query_keywords = {
            "소프트웨어학과 졸업요건을 알려줘": {
                "title_keywords": ["졸업", "졸업요건", "졸업사정", "학사"],
                "content_keywords": ["졸업", "이수", "학점", "필수과목", "소프트웨어", "컴퓨터"],
                "dept_weights": {
                    "cse": 0.8,
                    "skku": 0.5,
                    "ai": 0.6,
                }
            },
            # 소프트웨어학과 줄임말인 솦에 대해서도 동일한 쿼리로 처리
            "솦 졸업요건을 알려줘": {
                "title_keywords": ["졸업", "졸업요건", "졸업사정", "학사"],
                "content_keywords": ["졸업", "이수", "학점", "필수과목", "소프트웨어", "컴퓨터"],
                "dept_weights": {
                    "cse": 0.8,
                    "skku": 0.5,
                    "ai": 0.6,
                }
            },
            "장학금 신청 방법 알려줘": {
                "title_keywords": ["장학", "장학금", "신청"],
                "content_keywords": ["장학", "신청방법", "지원", "서류", "제출"],
                "dept_weights": {
                    "skku": 0.8,
                    "cse": 0.6,
                    "physics" : 0.6,
                    "ai": 0.6,
                    "biz": 0.6,
                }
            },
            "기숙사 신청 언제해?": {
                "title_keywords": ["기숙사", "생활관", "입사", "신관", "인관", "의관", "예관", "지관"],
                "content_keywords": ["기숙사", "생활관", "신청기간", "모집", "선발", "신관", "인관", "의관", "예관", "지관"],
                "dept_weights": {
                    "dorm": 0.9,
                    "skku": 0.4,
                }
            },
            "인공지능학과 관련 채용 연계 공지사항": {
                "title_keywords": ["인공지능", "ai", "채용", "연계", "공고", "공지"],
                "content_keywords": ["모집", "채용", "연계", "공고", "공지", "인공지능", "ai", "인재", "인력"],
                "dept_weights": {
                    "ai": 0.8,
                    "skku": 0.5,
                }
            },
            "물리학과 관련 대회가 있어?": {
                "title_keywords": ["경진대회", "대회", "공모전", "공모", "대학생", "학술대회"],
                "content_keywords": ["경진대회", "대회", "공모전", "공모", "대학생", "학술대회", "물리학", "물리", "모집", "참가", "지원", "신청"],
                "dept_weights": {
                    "physics": 0.8,
                    "skku": 0.4,
                }
            },
        }

    def calculate_content_similarity_score(self, text: str, keywords: list) -> float:
        # 키워드 기반 컨텐츠 유사도 점수 계산
        text_lower = text.lower()
        keyword_count = sum(1 for keyword in keywords if keyword in text_lower)
        weight = 1.2
        return min(1.0, keyword_count / len(keywords) * weight)

    def create_query_based_examples(self, df: pd.DataFrame) -> List[InputExample]:
        training_examples = []
        
        for query, criteria in self.query_keywords.items():
            title_keywords = criteria["title_keywords"]
            content_keywords = criteria["content_keywords"]
            dept_weights = criteria["dept_weights"]
            
            # 각 공지사항에 대해 종합적인 점수 계산
            for _, notice in df.iterrows():
                # 1. name 가중치 계산
                dept_score = dept_weights.get(notice['name'].lower(), 0.1)
                
                # 2. title 유사도 계산
                title_score = self.calculate_content_similarity_score(
                    notice['title'], 
                    title_keywords
                )
                
                # 3. content 유사도 계산
                content_score = self.calculate_content_similarity_score(
                    notice['content'], 
                    content_keywords
                )
                
                # total score (dept 40%, title 35%, content 25%)
                final_score = (
                    dept_score * 0.4 + 
                    title_score * 0.35 + 
                    content_score * 0.25
                )
                
                # score >= 0.2 경우만 학습 데이터에 포함
                if final_score >= 0.2:
                    training_examples.append(
                        InputExample(
                            texts=[
                                query,
                                f"{notice['title']} {notice['content']}"
                            ],
                            label=final_score
                        )
                    )

        return training_examples

    def create_content_based_examples(self, df: pd.DataFrame) -> List[InputExample]:
        training_examples = []
        
        notices = df.to_dict('records')
        n = len(notices)
        
        # 1. 제목 기반 유사도
        for i in range(n):
            for j in range(i + 1, n):
                row1 = notices[i]
                row2 = notices[j]
                
                # 제목의 키워드 중복도 계산
                title1_words = set(row1['title'].lower().split())
                title2_words = set(row2['title'].lower().split())
                
                if len(title1_words | title2_words) > 0:
                    title_similarity = len(title1_words & title2_words) / len(title1_words | title2_words)
                    
                    if title_similarity > 0.3:
                        training_examples.append(
                            InputExample(
                                texts=[
                                    f"{row1['title']} {row1['content']}", 
                                    f"{row2['title']} {row2['content']}"
                                ],
                                label=0.7 + title_similarity * 0.3
                            )
                        )

        # 2. 동일 카테고리, 시간 순서 공지사항
        for category in df['category'].unique():
            category_notices = df[df['category'] == category].sort_values('notice_date').to_dict('records')
            
            for i in range(len(category_notices)-1):
                training_examples.append(
                    InputExample(
                        texts=[
                            f"{category_notices[i]['title']} {category_notices[i]['content']}", 
                            f"{category_notices[i+1]['title']} {category_notices[i+1]['content']}"
                        ],
                        label=0.6
                    )
                )

        return training_examples
    
    def create_combined_training_examples(self, df: pd.DataFrame) -> List[InputExample]:
        training_examples = []
        
        # 1. 질문-공지사항
        query_examples = self.create_query_based_examples(df)
        training_examples.extend(query_examples)
        
        # 2. content keyword
        content_examples = self.create_content_based_examples(df)
        training_examples.extend(content_examples)
        
        # 3. random negative 예제
        num_negative = len(training_examples) // 4  # 전체 예제의 25%
        for _ in range(num_negative):
            random_rows = df.sample(n=2)
            if random_rows.iloc[0]['category'] != random_rows.iloc[1]['category']:
                training_examples.append(
                    InputExample(
                        texts=[
                            f"{random_rows.iloc[0]['title']} {random_rows.iloc[0]['content']}", 
                            f"{random_rows.iloc[1]['title']} {random_rows.iloc[1]['content']}"
                        ],
                        label=0.1
                    )
                )
        
        return training_examples