#!/usr/bin/env python

import sys,time

help_str = 'The script prints the best score and iteration given the caffe log file \n' + \
            'Arguments: \n' + \
            '1st - filename \n' + \
            '2nd - time interval of printing (if < 0 - print only once)'

if len(sys.argv) < 2:
    print help_str
    exit()

in_file = ''
argpos = 1
if len(sys.argv) > argpos:
    in_file = sys.argv[argpos]  

time_int = -1.
argpos = 2
if len(sys.argv) > argpos:
    time_int = float(sys.argv[argpos])

acc_str = 'accuracy ='    
iter_str = 'Iteration'

print 'File', in_file, 'Interval:',time_int

while True:
    accuracies = []
    f = open(in_file)
    lines = f.readlines()
    line_indx = -1
    print 'Lines total', len(lines)
    for line in lines:
        line_indx = line_indx + 1
        pos = line.find(acc_str)
        if pos > 0:
            acc_cur = -1
            acc_str_cur = line[pos:]
            acc_str_cur = acc_str_cur[ (acc_str_cur.find('=') + 1): ]
            acc_cur = float( acc_str_cur )
            
            #print acc_str_cur            
            
            #Processing the iteration string
            iter_cur = -1
            iter_line = lines[line_indx - 1]            
            iter_pos_start = iter_line.find( iter_str )
            iter_pos_end = iter_line.find( ',' )
            if iter_pos_start >= 0 and iter_pos_end > 0:
                iter_pos_start = iter_pos_start + len(iter_str)
            if iter_pos_start >=0 and iter_pos_end > 0 and iter_pos_start < len(iter_line) and iter_pos_end < len(iter_line):
                iter_str_cur = iter_line[iter_pos_start:iter_pos_end]                
                #print iter_line, iter_pos_start, iter_pos_end, iter_line[iter_pos_end]                
                iter_cur = float( iter_str_cur )
            
            if(acc_cur >= 0 and iter_cur >= 0):
                accuracies.append([iter_cur, acc_cur])
    
    #Finding best
    max_it_indx = 0;
    for cur_it_indx in range(len(accuracies)):
        #print 'Acc = ', accuracies[cur_it_indx][1], ' It = ', accuracies[cur_it_indx][0]
        if accuracies[max_it_indx][1] < accuracies[cur_it_indx][1] :
            max_it_indx = cur_it_indx
    
    #Printing results
    if len(accuracies) > 0:
        print 'Max accuracy = ', accuracies[max_it_indx][1], ' Iteration = ', accuracies[max_it_indx][0]
            
    if time_int < 0:
        break
    else:
        time.sleep(time_int)
    
