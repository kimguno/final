import sys
sys.path.append('C:/big18/final/test/DB/')
import DBconnection as db
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime, timedelta

def get_vector_list(code, num, set_num):
    data = db.execute_query(code, num ,set_num)
    data.sort_values(['Jdate'],ascending=1, inplace=True)
    print(data['Jdate'][-1:])
    df = data[['code','종가','매수량','매도량']]
    close_price = data[['Jdate','종가']]

    def divide_volumes(row):
        try:
            divisor = db.get_stock_share(code)
            row['power'] = row['매수량'] - row['매도량']
            row['power'] = row['power']*1000000 / divisor
        except KeyError:
            print(f"코드 {row['code']}를 찾지 못했습니다.")
            input()
        return row

    df = divide_volumes(df)

    # 데이터 로드
    df=df[['code','종가', 'power']]
    df.rename(columns={'종가' : 'close'}, inplace=True)

    # 결과를 저장할 리스트
    results = []

    for i in range(num-9):
        # 벡터화하여 연산
        vectorList = []
        diff = (df['close'].values[i:i+9] - df['close'].values[i+1:i+10]) / df['close'].values[i:i+9] * 10000
        vectorList.extend(round(d, 4) for d in diff)
        diff = (df['power'].values[i:i+9] - df['power'].values[i+1:i+10])
        vectorList.extend(round(d, 2) for d in diff)

        results.append(vectorList)

    # 결과를 DataFrame으로 변환
    results_df = pd.DataFrame(results)
    return results_df, close_price
