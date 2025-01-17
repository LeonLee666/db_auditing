import random
import threading
import pymysql
import argparse
import numpy as np
import sys
from ratelimiter import RateLimiter  # 添加这行导入

rate_limit = 10
zipf = 1.2
def get_connection():
    """获取数据库连接"""
    return pymysql.connect(
        host='127.0.0.1',
        user='root',
        password='123456',
        database='tpcc',
        charset='utf8mb4',
        connect_timeout=5,
        read_timeout=30,
    )

def generate_s_i_id(dist_type='uniform'):
    """根据指定分布生成s_i_id"""
    if dist_type == 'uniform':
        return np.random.randint(1, 100001)
    elif dist_type == 'zipf':
        # Zipf分布，alpha=1.5表示相对陡峭的幂律分布
        return int(np.random.zipf(zipf, size=1)[0] % 100000) + 1
    elif dist_type == 'normal':
        # 正态分布，均值设在50000，标准差为16666
        value = int(np.random.normal(50000, 16666))
        # 确保值在1-100000范围内
        return max(1, min(100000, value))
    return np.random.randint(1, 100001)  # 默认返回均匀分布

def query_as_client(dist_type='uniform'):
    try:
        connection = get_connection()
        cursor = connection.cursor()
        thread_rate_limiter = RateLimiter(max_calls=rate_limit, period=1)  # 每个线程独立的限流器
        
        for _ in range(500000):
            try:
                with thread_rate_limiter:  # 使用线程独立的限流器
                    s_w_id = random.randint(1, 5)
                    s_i_id = generate_s_i_id(dist_type)
                    query = f"SELECT * FROM bmsql_stock WHERE s_w_id = {s_w_id} AND s_i_id = {s_i_id};"
                    cursor.execute(query)
                    results = cursor.fetchall()
                    connection.commit()
            except pymysql.MySQLError as err:
                connection.rollback()
                print(f"Error: {err}")
        cursor.close()
        connection.close()
    except Exception as e:
        print(f"Outer error: {e}")

def query_as_bot(dist_type='uniform'):
    try:
        connection = get_connection()
        cursor = connection.cursor()
        thread_rate_limiter = RateLimiter(max_calls=rate_limit, period=1)  # 添加线程独立的限流器
        
        for s_w_id in range(5, 0, -1):
            for s_i_id in range(100000, 0, -1):
                try:
                    with thread_rate_limiter:  # 使用限流器控制查询速率
                        query = f"SELECT * FROM bmsql_stock WHERE s_w_id = {s_w_id} AND s_i_id = {s_i_id};"
                        cursor.execute(query)
                        results = cursor.fetchall()
                        connection.commit()
                except pymysql.MySQLError as err:
                    connection.rollback()
                    print(f"Error: {err}")
        cursor.close()
        connection.close()        
        sys.exit(0)
    except Exception as e:
        print(f"Outer error: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='数据库查询测试工具')
    parser.add_argument('-c', '--client', 
                       type=int, 
                       default=10,
                       help='普通查询线程数量 (默认: 10)')
    parser.add_argument('-b', '--query_as_bot', 
                       type=int, 
                       default=1,
                       help='爬虫查询线程数量 (默认: 1)')
    parser.add_argument('-d', '--distribution',
                       type=str,
                       choices=['uniform', 'zipf', 'normal'],
                       default='uniform',
                       help='s_i_id的数据分布类型 (默认: uniform)')
    
    args = parser.parse_args()
    threads = []
    
    # 创建普通查询线程
    for _ in range(args.client):
        thread = threading.Thread(target=query_as_client, args=(args.distribution,))
        threads.append(thread)
        thread.start()
    
    # 创建爬虫查询线程
    for _ in range(args.query_as_bot):
        query_as_bot_thread = threading.Thread(target=query_as_bot, args=(args.distribution,))
        threads.append(query_as_bot_thread)
        query_as_bot_thread.start()
        
    # 等待所有线程完成
    for thread in threads:
        thread.join()
