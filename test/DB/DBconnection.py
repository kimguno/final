import pymysql
import pandas as pd

def get_stock_map_load():
    data = pd.read_csv('C:/big18/dl-dev/dl-dev/project/전종목시세.csv',encoding='cp949')
    data =data[['종목코드','종목명','상장주식수']]
    result_map = {row['종목코드']: {'종목명': row['종목명'], '상장주식수': row['상장주식수']} for index, row in data.iterrows()}
    return result_map

def get_stock_share(code):
    map = get_stock_map_load()
    stock_share = map[f'{code}']['상장주식수']
    return stock_share

def get_stock_name(code):
    map = get_stock_map_load()
    stock_name = map[f'{code}']['종목명']
    return stock_name



###################################################디비연동#########################################

def connection():
    connection = pymysql.connect(
        host='192.168.40.53',       # 데이터베이스 호스트
        user='root',                # 사용자 이름
        password='big185678',       # 비밀번호
        database='finaldb',         # 데이터베이스 이름
    )
    return connection

# def select_sql(code, num):
#     query = f"""
#             SELECT am.code 
#                   , am.close_price 
#                   , am.sell_vol 
#                   , am.buy_vol 
#                   , am.Jdate 
#               FROM a{code}_mindata am
#               ORDER BY am.Jdate DESC
#               LIMIT {num};
#             """
#     return query

# 테스트하려고 만들었다 시발롬아
def select_sql(code, num, set_num):
    query = f"""
            SELECT am.code 
                  , am.close_price 
                  , am.sell_vol 
                  , am.buy_vol 
                  , am.Jdate 
              FROM a{code}_mindata am
              where am.Jdate > '2024-09-05'
              ORDER BY am.Jdate ASC
              LIMIT {num} OFFSET {set_num};
            """
    return query

def column_rename(df):
    df.rename(columns={'close_price': '종가', 'sell_vol': '매도량', 'buy_vol': '매수량'}, inplace=True)

def execute_query(code,num,set_num):
    conn = connection()
    if conn is None:
        return None
    
    try:
        query = select_sql(code, num, set_num)
        with conn.cursor() as cursor:
            cursor.execute(query)
            result = cursor.fetchall()  # 모든 결과를 가져옵니다.
            columns = [column[0] for column in cursor.description]
            result_df = pd.DataFrame(result, columns=columns)
            column_rename(result_df)
            return result_df
    except pymysql.MySQLError as e:
        print(f"Error executing query: {e}")
        return None
    finally:
        conn.close()  # 커넥션 종료







