import random
import threading
import pymysql
from dbutils.pooled_db import PooledDB
from tqdm import tqdm
import argparse

# 修改连接池配置
pool = PooledDB(
    creator=pymysql,
    maxconnections=50,  # 可以适当增加最大连接数
    mincached=10,
    maxcached=20,
    maxshared=20,
    blocking=True,
    host='127.0.0.1',
    user='root',
    password='123456',
    database='tpcc',
    charset='utf8mb4',
    connect_timeout=5,  # 添加连接超时设置
    read_timeout=30,    # 添加读取超时设置
)

def query_as_normal():
    try:
        # 在循环外获取连接
        connection = pool.connection()
        cursor = connection.cursor()
        
        for _ in tqdm(range(500000), desc="Querying Normal"):
            try:
                s_w_id = random.randint(1, 5)
                s_i_id = random.randint(1, 100000)
                query = f"SELECT * FROM bmsql_stock WHERE s_w_id = {s_w_id} AND s_i_id = {s_i_id};"
                cursor.execute(query)
                results = cursor.fetchall()
                connection.commit()
            except pymysql.MySQLError as err:
                connection.rollback()
                print(f"Error: {err}")
        # 循环结束后关闭连接
        cursor.close()
        connection.close()
    except Exception as e:
        print(f"Outer error: {e}")

def query_as_bot():
    try:
        # 在循环外获取连接
        connection = pool.connection()
        cursor = connection.cursor()
        
        for s_w_id in tqdm(range(5, 0, -1), desc="Querying query_as_bot"):
            for s_i_id in tqdm(range(100000, 0, -1), desc="Querying query_as_bot/Items", leave=False):
                try:
                    query = f"SELECT * FROM bmsql_stock WHERE s_w_id = {s_w_id} AND s_i_id = {s_i_id};"
                    cursor.execute(query)
                    results = cursor.fetchall()
                    connection.commit()
                except pymysql.MySQLError as err:
                    connection.rollback()
                    print(f"Error: {err}")
        # 循环结束后关闭连接
        cursor.close()
        connection.close()
    except Exception as e:
        print(f"Outer error: {e}")

if __name__ == '__main__':
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='数据库查询测试工具')
    parser.add_argument('-n', '--normal', 
                       type=int, 
                       default=10,
                       help='普通查询线程数量 (默认: 10)')
    parser.add_argument('-b', '--query_as_bot', 
                       type=int, 
                       default=1,
                       help='爬虫查询线程数量 (默认: 1)')
    
    args = parser.parse_args()
    threads = []
    
    # 创建普通查询线程
    for _ in range(args.normal):
        thread = threading.Thread(target=query_as_normal)
        threads.append(thread)
        thread.start()
    
    # 创建爬虫查询线程
    for _ in range(args.query_as_bot):
        query_as_bot_thread = threading.Thread(target=query_as_bot)
        threads.append(query_as_bot_thread)
        query_as_bot_thread.start()
        
    # 等待所有线程完成
    for thread in threads:
        thread.join()
