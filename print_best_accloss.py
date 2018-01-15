#!/usr/bin/env python

import sys,time
import re
import matplotlib.pyplot as plt
import numpy as np

## Structure of the testing
#Iteration 100, Testing net (#0)
#Test loss: 4.86363
#    Test net output #0: bb-loss = 1.44553 (* 3 = 4.3366 loss)
#    Test net output #1: pixel-acc = 0.942303
#    Test net output #2: pixel-loss = 0.216342 (* 1 = 0.216342 loss)
#    Test net output #3: type-acc = 0.934385
#    Test net output #4: type-loss = 0.310691 (* 1 = 0.310691 loss)

## Searching regexp
# re.search('\s[0-9]+\,', bla4).group(0)
# re.search('\s[0-9]+\.[0-9]+[\s]*', lines[1]).group(0)

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

def header_check(line, name):
    pos = line.find(name)
    val = None
    if pos > 0:
        match = re.search('\s[0-9]+\,', line).group(0)
        match = match.rstrip(',')
        match = match.rstrip('\n')
        match = match.rstrip(' ')
        match = match.lstrip(' ')
        val = int(match)
    return (pos,val)

def loss_check(line, name):
    pos = line.find(name)
    val = None
    if pos > 0:
        match = re.search('\s[0-9]+\.[0-9]+[\s]*', line).group(0)
        match = match.rstrip(' ')
        match = match.rstrip('\n')
        match = match.lstrip(' ')
        val = float(match)
    return (pos,val)

iter_str = 'Iteration'
proc_key  = ['Testing net', 'Test loss', 'bb-loss', 'pixel-acc', 'pixel-loss', 'type-acc', 'type-loss']
proc_func = [header_check, loss_check, loss_check, loss_check, loss_check, loss_check, loss_check]
proc_minmax = [key.find('loss') >=0 for key in proc_key]
proc_fig = [plt.figure(i) for i in range(len(proc_key)-1)]


print 'File', in_file, 'Interval:', time_int

updates = 0


while True:
    updates += 1
    proc_step = 0
    proc_val = [[] for i in range(len(proc_key))]
    f = open(in_file)
    lines = f.readlines()
    print 'Update: %d Lines total %d' % (updates, len(lines))
    for line_indx, line in enumerate(lines):
        pos,val = proc_func[proc_step](line, proc_key[proc_step])
        if pos > 0:
            proc_val[proc_step].append(val)
            proc_step += 1
            if proc_step >= len(proc_key):
                proc_step = 0
        
    for i, key in enumerate(proc_key):
        if i == 0:
            continue
        #print('key:', proc_key[i], ' values:\n', proc_val[i])
        
        plt.figure(i-1)
        plt.clf()
        plt.plot(proc_val[0], proc_val[i])
        if proc_minmax[i]:
            best_id = np.argmin(proc_val[i])
            best_val = np.min(proc_val[i])
        else:
            best_id = np.argmax(proc_val[i])
            best_val = np.max(proc_val[i])
        plt.title(str(proc_key[i]) + ':: val_{best}: ' + str(best_val) + ' i_{best}: ' + str(proc_val[0][best_id]))
        plt.pause(0.01)
    plt.show(block=False)
    
    #Printing results
    #if len(accuracies) > 0:
    #    print 'Max accuracy = ', accuracies[max_it_indx][1], ' Iteration = ', accuracies[max_it_indx][0]
            
    if time_int < 0:
        break
    else:
        time.sleep(time_int)
    
