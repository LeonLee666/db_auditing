mysql-reboot() {
	mycli -h 127.0.0.1 -P 3306 -u root -p123456 -e "shutdown"
	sleep  3 
	mysqld --defaults-file=/home/liliang/mysql/my.cnf
}

rm ~/mysql/data/audit*

client_nums=(10 50 100 200 300 400 500 1000)

# 循环处理每个 alpha 值
for num in "${client_nums[@]}"; do
	# 测试 b=0 的情况
	mysql-reboot
	python3 tpcc.py -c $num -b 0 -d uniform -r $((5000/num))
	mv ~/mysql/data/audit_log.2025-01-21.0 "uniform/no${num}.log"
	echo 生成正样例
	# 测试 b=1 的情况
	mysql-reboot
	python3 tpcc.py -c $((num-1)) -b 1 -d uniform -r $((5000/num))
	mv ~/mysql/data/audit_log.2025-01-21.0 "uniform/yes${num}.log"
done

# 定义要测试的 alpha 值数组
alphas=(1.2 1.4 1.6 1.8 2.0)

# 循环处理每个 alpha 值
for alpha in "${alphas[@]}"; do
	# 测试 b=0 的情况
	mysql-reboot
	python3 tpcc.py -c 100 -b 0 -d zipf -a $alpha
	mv ~/mysql/data/audit_log.2025-01-17.0 "zipf100/no${alpha}.log"

	# 测试 b=1 的情况
	mysql-reboot
	python3 tpcc.py -c 99 -b 1 -d zipf -a $alpha
	mv ~/mysql/data/audit_log.2025-01-17.0 "zipf100/yes${alpha}.log"
done

for alpha in "${alphas[@]}"; do
	# 测试 b=0 的情况
	mysql-reboot
	python3 tpcc.py -c 500 -b 0 -d zipf -a $alpha
	mv ~/mysql/data/audit_log.2025-01-17.0 "zipf500/no${alpha}.log"

	# 测试 b=1 的情况
	mysql-reboot
	python3 tpcc.py -c 499 -b 1 -d zipf -a $alpha
	mv ~/mysql/data/audit_log.2025-01-17.0 "zipf500/yes${alpha}.log"
done