import pandas as pd

# 예시: 전문가 CSV 불러오기
# expert_df = pd.read_csv("gold_standardized _doublecheck.csv", encoding='euc-kr')  # 전문가가 작성한 신약명 CSV
llm_df = pd.read_csv("/data1/workspace/prepro_calcul/t3.csv",encoding='euc-kr')        # LLM에서 추출한 신약명 CSV

# 약물명 정규화 함수
def normalize_drug_list(drug_str):
    if pd.isna(drug_str):
        return []
    # ';'로 분리
    drugs = drug_str.split(';')
    # '-' 제거, 공백 제거, 소문자 변환
    drugs = [d.replace('-', '').replace(' ', '').lower() for d in drugs if d.strip() != '']
    return drugs

# 전문가 CSV, LLM CSV에 적용
# expert_df['normalized'] = expert_df['drug_column'].apply(normalize_drug_list)
llm_df['normalized'] = llm_df['drug'].apply(normalize_drug_list)