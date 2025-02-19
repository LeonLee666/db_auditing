import os
import random
import threading
import time
import pymysql
import argparse
import numpy as np
import sys
from ratelimiter import RateLimiter  # 添加这行导入

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

def generate_s_i_id(dist_type='uniform', alpha=1.2):
    """根据指定分布生成s_i_id"""
    if dist_type == 'uniform':
        return np.random.randint(1, 100001)
    elif dist_type == 'zipf':
        # Zipf分布，使用传入的alpha参数
        return int(np.random.zipf(alpha, size=1)[0] % 100000) + 1
    elif dist_type == 'normal':
        # 正态分布，均值设在50000，标准差为16666
        value = int(np.random.normal(50000, 16666))
        # 确保值在1-100000范围内
        return max(1, min(100000, value))
    return np.random.randint(1, 100001)  # 默认返回均匀分布

def query_as_client(dist_type='uniform', alpha=1.2):
    try:
        connection = get_connection()
        cursor = connection.cursor()
        thread_rate_limiter = RateLimiter(max_calls=rate_limit, period=1)
        
        for _ in range(500000):
            try:
                with thread_rate_limiter:
                    s_w_id = random.randint(1, 5)
                    s_i_id = generate_s_i_id(dist_type, alpha)
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
        thread_rate_limiter = RateLimiter(max_calls=rate_limit, period=1)
        
        for s_w_id in range(5, 0, -1):
            for s_i_id in range(100000, 0, -1):
                try:
                    with thread_rate_limiter:
                        query = f"SELECT * FROM bmsql_stock WHERE s_w_id = {s_w_id} AND s_i_id = {s_i_id};"
                        cursor.execute(query)
                        results = cursor.fetchall()
                        connection.commit()
                except pymysql.MySQLError as err:
                    connection.rollback()
                    print(f"Error: {err}")
        cursor.close()
        connection.close()        
        os._exit(0)
    except Exception as e:
        print(f"Outer error: {e}")
        os._exit(1)

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
    parser.add_argument('-a', '--alpha',
                       type=float,
                       default=1.2,
                       help='Zipf分布的alpha参数 (默认: 1.2)')
    parser.add_argument('-r', '--rate_limit',
                       type=int,
                       default=10,
                       help='每秒最大查询次数限制 (默认: 500)')
    
    args = parser.parse_args()
    # 设置全局rate_limit变量
    global rate_limit
    rate_limit = args.rate_limit
    threads = []
    
    # 创建普通查询线程
    for _ in range(args.client):
        thread = threading.Thread(target=query_as_client, args=(args.distribution, args.alpha))
        threads.append(thread)
        thread.start()
    
    # 创建爬虫查询线程
    for _ in range(args.query_as_bot):
        query_as_bot_thread = threading.Thread(target=query_as_bot, args=(args.distribution,))
        threads.append(query_as_bot_thread)
        query_as_bot_thread.start()
    # 监控审计日志文件大小
    while True:
        try:
            file_size = os.path.getsize('/home/liliang/mysql/data/audit_log.2025-01-21.0')
            if file_size > 600 * 1024 * 1024:  # 600MB
                print("审计日志文件大小超过600MB,程序退出")
                os._exit(0)
            time.sleep(1)  # 每秒检查一次
        except FileNotFoundError:
            print("未找到审计日志文件")
            break
        except Exception as e:
            print(f"监控文件大小时发生错误: {e}")
            break
    # 等待所有线程完成
    for thread in threads:
        thread.join()
