for method in centroid grid euler; do
	./generate_config.sh 10000 "4096" $method
    for client_nums in 10 50 100 500 1000; do
        python3 kickoff.py --cuda --positive uniform/yes${client_nums}.log --negative uniform/no${client_nums}.log
    done
done