本项目是《DBAuditGaurd》论文的实验评估代码

# 运行方法

## 生成数据集

```
# 下面的命令会产生100个线程，每个线程每秒进行50次的查询。其中有99个线程正常查询，一个线程是bot线程
python3 tpcc.py -c 99 -b 1 -d uniform -r 50
```

## 算法评估

```
# 启动kickoff脚本，测试正负两个sql脚本
python3 kickoff.py --cuda --positive file1.log --negative file2.log
```
