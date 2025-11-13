import pandas as pd

# 1. CSV 파일 불러오기
df = pd.read_csv('gold_single.csv')

# 2. 문자열이 포함된 열을 소문자로 변환
# 'column_name'은 변환하려는 열의 이름으로 바꾸세요.
# df['single_drugs_str'] = df['single_drugs_str'].str.lower()

# # 3. 전체 문자열 열에 대해 소문자 변환 적용
# # df.select_dtypes(include='object').columns는 모든 문자열 열을 선택합니다.
# for col in df.select_dtypes(include='object').columns:
#     df[col] = df[col].str.lower()

# # 4. 변환된 데이터를 새로운 CSV 파일로 저장
# df.to_csv('gold_converted.csv', index=False)

#---------------------------------------------------------------
import pandas as pd
from drugname_standardizer import DrugStandardizer
import os
import sys

# =========================================================================
# 📌 설정 변수 (사용자의 파일 정보에 맞게 수정하세요)
# =========================================================================
# 1. 입력 CSV 파일 이름
INPUT_FILE_NAME = 'gold_converted.csv' 
# 2. 약물 이름이 들어있는 컬럼 이름
DRUG_NAME_COLUMN = 'single_drugs_str' 
# 3. 출력할 새로운 CSV 파일 이름
OUTPUT_FILE_NAME = 'standardized_drugs.csv'
# =========================================================================


import pandas as pd
from drugname_standardizer import DrugStandardizer

# =========================================================================
# 📌 설정 변수 (사용자의 파일 정보에 맞게 수정하세요)
# =========================================================================
# 1. 입력 CSV 파일 이름
INPUT_FILE_NAME = 'gold_converted.csv' 
# 2. 약물 이름이 들어있는 컬럼 이름
DRUG_LIST_COLUMN = 'single_drugs_str' 
# 3. 약물 리스트를 구분하는 기호 (여기서는 세미콜론)
DELIMITER = ';'
# 4. 출력할 새로운 CSV 파일 이름
OUTPUT_FILE_NAME = 'gold_standardized.csv'
# =========================================================================


def standardize_drug_list_cell(drug_list_string, standardizer):
    """
    세미콜론으로 구분된 약물 이름 문자열을 받아 각 항목을 표준화 후 다시 결합합니다.
    """
    if pd.isna(drug_list_string) or not drug_list_string:
        return drug_list_string  # 빈 값은 그대로 반환

    # 1. 세미콜론(;)을 기준으로 각 약물 이름 분리
    # 공백을 제거(.strip())하여 이름만 깔끔하게 추출
    drug_names = [name.strip() for name in str(drug_list_string).split(DELIMITER)]
    
    # 2. 각 약물 이름을 표준화 (브랜드 -> 제네릭)
    standardized_names = []
    for name in drug_names:
        if name: # 이름이 비어있지 않은 경우에만 표준화 시도
            # standardizer.standardize_name() 사용
            generic_name = standardizer.standardize_name(name)
            standardized_names.append(generic_name)
        
    # 3. 표준화된 이름들을 다시 세미콜론(;)으로 결합
    return DELIMITER.join(standardized_names)


def process_csv_standardization(input_file, drug_list_column, output_file):
    """
    CSV 파일을 처리하고 표준화된 결과를 저장하는 메인 함수
    """
    try:
        # 1. 데이터 불러오기
        df = pd.read_csv(input_file)
        print(f"✅ 파일 로드 성공: {input_file} ({len(df)}개 항목)")

        if drug_list_column not in df.columns:
            print(f"❌ 오류: 컬럼 '{drug_list_column}'을 파일에서 찾을 수 없습니다. 컬럼 이름을 확인해주세요.")
            return

        # 2. 표준화 도구 초기화
        print("🔧 약물 표준화 도구 초기화 중...")
        standardizer = DrugStandardizer()
        
        # 3. 표준화 컬럼 생성 및 함수 적용
        print("🚀 약물 리스트 표준화 시작...")
        
        # apply 함수를 사용하여 각 셀(세미콜론 리스트)에 대해 standardize_drug_list_cell 함수 적용
        df['Generic_Drug_List'] = df[drug_list_column].apply(
            lambda x: standardize_drug_list_cell(x, standardizer)
        )
        
        print("✅ 표준화 완료!")

        # 4. 결과 저장
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"💾 결과 저장 성공: {output_file}")
        
        # 표준화 전후 예시 출력
        print("\n--- 표준화 결과 미리보기 (상위 5개) ---")
        print(df[[drug_list_column, 'Generic_Drug_List']].head())

    except FileNotFoundError:
        print(f"❌ 오류: 파일을 찾을 수 없습니다. '{input_file}' 파일이 현재 디렉토리에 있는지 확인해주세요.")
    except Exception as e:
        print(f"❌ 예기치 않은 오류 발생: {e}")


if __name__ == "__main__":
    process_csv_standardization(INPUT_FILE_NAME, DRUG_LIST_COLUMN, OUTPUT_FILE_NAME)



#---------------------------------
#소문자 재변환
# 1. CSV 파일 불러오기
df = pd.read_csv('gold_standardized.csv')

# 2. 문자열이 포함된 열을 소문자로 변환
# 'column_name'은 변환하려는 열의 이름으로 바꾸세요.
df['Generic_Drug_List'] = df['Generic_Drug_List'].str.lower()

# 3. 전체 문자열 열에 대해 소문자 변환 적용
# df.select_dtypes(include='object').columns는 모든 문자열 열을 선택합니다.
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].str.lower()

# 4. 변환된 데이터를 새로운 CSV 파일로 저장
df.to_csv('gold_standardized.csv', index=False)
