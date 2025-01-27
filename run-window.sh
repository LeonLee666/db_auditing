for method in centroid grid euler; do
    for win_size in 128 256 512 1024 2048 4096 8192; do
        ./generate_config.sh 10000 "$win_size" $method
        python3 kickoff.py --cuda --positive uniform/yes100.log --negative uniform/no100.log
    done
done