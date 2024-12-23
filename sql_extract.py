from multiprocessing import Pool
import sqlparse
from sqlparse.sql import TokenList
from sqlparse.tokens import Number, String, Literal
import pandas as pd
import json
from tqdm import tqdm

def read_json_file(input_file):
    # 初始化一个空列表来存储解析后的数据
    data = []

    # 读取文件并解析 JSON
    with open(input_file, 'r') as file:
        for line in file:
            # 去掉行末尾的换行符并解析为字典
            record = json.loads(line.strip())
            data.append(record)
            
    # 创建 DataFrame
    df = pd.DataFrame(data)
    # 提取 'sql' 字段
    sql_df = df[['sql']]
    
    return sql_df


class SQLExtract:
    def __init__(self):
        # 扩展需要替换的字面量类型
        self.literals = (String.Single, String.Double, Number.Integer,
                         Number.Float, String.Symbol, Literal)
        # 添加一个列表来存储被替换的字面量
        self.replaced_literals = []

    def _process_token(self, token):
        # 检查token类型是否在需要替换的类型中
        if token.ttype in self.literals or (isinstance(token.ttype, tuple) and
                                            any(t in self.literals for t in token.ttype)):
            # 存储被替换的字面量值
            self.replaced_literals.append(token.value)
            return ' ? '
        # 如果是TokenList，递归处理
        elif isinstance(token, TokenList):
            return ''.join(self._process_token(t) for t in token.tokens)
        # 其他类型的token保持不变
        else:
            return str(token)

    def extract_template(self, sql_text):
        # 重置字面量列表
        self.replaced_literals = []

        # 解析SQL语句
        parsed = sqlparse.parse(sql_text)

        result = []
        # 处理每个SQL语句
        for statement in parsed:
            # 格式化SQL以确保一致的空格
            formatted_sql = sqlparse.format(str(statement), reindent=False, strip_whitespace=True)
            # 重新解析格式化后的SQL
            reparsed = sqlparse.parse(formatted_sql)[0]
            # 递归处理每个token
            processed_sql = ''.join(self._process_token(token) for token in reparsed.tokens)
            result.append(processed_sql.strip())

        # 返回处理后的SQL语句
        return ''.join(result)

    def extract_literals(self):
        """
        返回所有被替换的字面量值
        """
        return self.replaced_literals

# 将 apply_extractor 定义在主模块级别
def apply_extractor(row):
    util = SQLExtract()  # 在每个进程中创建独立的 SQLExtract 对象
    return util.extract_template(row['sql']), util.extract_literals()

def parallel_apply_extractor(data, num_cpus):
    # 将 DataFrame 转换为列表
    data_list = data.to_dict('records')
    # 创建进程池
    with Pool(processes=num_cpus) as pool:
        results = pool.map(apply_extractor, data_list)
    # 将结果转换为 DataFrame
    result_df = pd.DataFrame(results, columns=['template', 'literals'])
    return result_df

def extract_sql_file(input_file):
    # 读取和处理 JSON 文件
    sql_df = read_json_file(input_file)
    
    # SQL模板化处理
    result_df = parallel_apply_extractor(sql_df, 40)
    sql_df['template'] = result_df['template']
    sql_df['literals'] = result_df['literals']
    
    utils = SQLExtract()
    filter_sql = utils.extract_template('SELECT * FROM bmsql_stock WHERE s_w_id = 5 AND s_i_id = 99998')
    sql_df = sql_df[sql_df['template'] == filter_sql]
    
    if len(sql_df) == 0:
        raise ValueError("No matching SQL template found")
    
    return sql_df


