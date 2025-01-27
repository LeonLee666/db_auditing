# 定义运行实验的函数
run_experiment() {
    local size=$1
    local method=$2
    ./generate_config.sh 10000 "4096" $method
    for alpha in 1.2 1.4 1.6 1.8 2.0; do
        python3 kickoff.py --cuda --positive zipf${size}/yes${alpha}.log --negative zipf${size}/no${alpha}.log
    done
}

# 运行所有实验组合
for size in 100 500; do
    for method in centroid grid euler; do
        run_experiment $size $method
    done
done