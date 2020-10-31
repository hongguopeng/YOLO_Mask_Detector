import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def extract_log(log_file , new_log_file , key_word):
    f = open(log_file , 'r')
    train_log = open(new_log_file , 'w')
    count = 0
    for line in f:
        if key_word in line and 'nan' not in line:
            train_log.write(line)
            count += 1
    f.close()
    train_log.close()
    return count


# Avg Loss
lines = extract_log('train_record.txt' , 'new_train_loss.txt' , 'avg')
names = ['Total Loss' , 'Avg Loss' , 'Learning Rate' , 'Total Time' , 'Training Images']
result_1 = pd.read_csv('new_train_loss.txt' ,
                       error_bad_lines = False ,
                       names = names)

result_1['Total Loss'] = result_1['Total Loss'].apply(lambda x : float(x.split(':')[1]))
result_1['Avg Loss'] = result_1['Avg Loss'].apply(lambda x : float(x.replace('avg' , '')))
result_1['Learning Rate'] = result_1['Learning Rate'].apply(lambda x : float(x.replace('rate' , '')))
result_1['Total Time'] = result_1['Total Time'].apply(lambda x : float(x.replace('seconds' , '')))
result_1['Training Images'] = result_1['Training Images'].apply(lambda x : float(x.replace('images' , '')))


# Avg IOU
lines = extract_log('train_record.txt' , 'new_train_IOU.txt' , 'IOU')
skiprows = [x for x in range(lines) if ((x % 10 != 9) |(x < 1000))]
names = ['Avg IOU' , 'Class' , 'Obj' , 'No Obj']
result_2 = pd.read_csv('new_train_IOU.txt' ,
                       header = None ,
                       skiprows = skiprows ,
                       error_bad_lines = False)
result_2 = result_2.iloc[: , :4]
result_2.columns = names

result_2['Avg IOU'] = result_2['Avg IOU'].apply(lambda x : float(x.split(':')[-1]))
result_2['Class'] = result_2['Class'].apply(lambda x : float(x.replace(' Class:' , '')))
result_2['Obj'] = result_2['Obj'].apply(lambda x : float(x.replace('Obj:' , '')))
result_2['No Obj'] = result_2['No Obj'].apply(lambda x : float(x.replace('No Obj:' , '')))


# 開始畫圖
fig , ax = plt.subplots(2 , 1 , figsize = (8 , 8))
ax[0].plot(result_1['Avg Loss'].values)
ax[0].set_title('Avg Loss Curve')
ax[1].plot(result_2['Avg IOU'].values)
ax[1].set_title('Avg IOU Curve')

