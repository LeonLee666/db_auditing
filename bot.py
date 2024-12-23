import random
import threading
import pymysql
import argparse
import numpy as np

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

def generate_s_i_id(dist_type='uniform', size=1):
    """根据指定分布生成s_i_id"""
    if dist_type == 'uniform':
        return random.randint(1, 100000)
    elif dist_type == 'zipf':
        # Zipf分布，alpha=1.5表示相对陡峭的幂律分布
        return int(np.random.zipf(1.5, size=1)[0] % 100000) + 1
    elif dist_type == 'normal':
        # 正态分布，均值设在50000，标准差为16666
        value = int(np.random.normal(50000, 16666))
        # 确保值在1-100000范围内
        return max(1, min(100000, value))
    return random.randint(1, 100000)  # 默认返回均匀分布

def query_as_normal(dist_type='uniform'):
    try:
        # 获取连接
        connection = get_connection()
        cursor = connection.cursor()
        
        for _ in range(500000):
            try:
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
        # 获取连接
        connection = get_connection()
        cursor = connection.cursor()
        
        for s_w_id in range(5, 0, -1):
            for s_i_id in range(100000, 0, -1):
                try:
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='数据库查询测试工具')
    parser.add_argument('-n', '--normal', 
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
    for _ in range(args.normal):
        thread = threading.Thread(target=query_as_normal, args=(args.distribution,))
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
