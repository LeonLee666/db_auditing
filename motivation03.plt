# 设置终端类型，这里以输出为png图片为例
set term post "Arial" eps color solid enh
set output "motivation03.eps"

set size 1.8,0.85
set boxwidth 0.4 


set tmargin 4
set bmargin 4
set lmargin 14
set rmargin 1

# 设置数据文件的格式，这里假设使用空格分隔符


# 设置柱状图的样式
set style data histograms
set style histogram clustered gap 2
set style fill solid border -1



# 设置 x 轴的标签

unset title
 
set yrange[0.5:1]
set ytics 0,0.2,1 font ",40"
set ylabel "Warm Invoc. Ratio" font ",40" offset -5,0,0


set xrange[-0.4:3.4]
set xtics ('No partition' 0, ' Trace B' 0.95, ' Trace C' 1.9, ' Trace D' 2.9) font ",40" offset character 0,-1.8,0
#set xtics nomirror
unset xlabel 
 
set boxwidth 0.9
# 将 set key 命令移到 plot 命令之前
set key reverse font 'Arial,38' spacing 1 outside maxrows 2 samplen 1 width -1.5

# 绘制柱状图和折线图
set datafile separator ","
plot 'motivation03.csv' using 2 title "TTL-based" fill pattern 1 border -1 linecolor 1, \
     'motivation03.csv' using 3 title "CH-RLU" fill pattern 6 border -1 linecolor rgb '#009EFF', \
     'motivation03.csv' using 4 title "Flame" fill pattern 2 border -1 linecolor rgb '#87C9C2', \
     'motivation03_line.csv' using 1:2 with linespoints pointtype 7 pointsize 1.5 linewidth 2 linecolor rgb '#FF0000' title "Line Plot"

set grid ytics


unset output
unset term