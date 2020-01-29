infile = open('toxref combined activity frequency.txt', 'r')
file = [i.rstrip() for i in infile.readlines()]
#print(len(file))

freq = {}
for item in file:
    if (item in freq):
        freq[item] += 1
    else:
        freq[item] = 1

for key in freq:
    print (key)

for value in freq:
    print(freq[value])

#
print(file[0])
count = 0
for item in range(0, len(file)):
    a = file[item]
    a = float(a)
    if -3 <= a and a < -2:
        count +=1
print(count)

# import pandas as pd
# file = pd.read_csv('toxref combined activity frequency.csv')
#
# count = 0
# active_freq = []
# inactive_freq = []
# for i in range(0, len(file)):
#
#     # print(file.iloc[i, 1])
#     if 1.32 >= file.iloc[i, 1] :
#         active_freq.append(file.iloc[i, 2])
#     if 1.32 < file.iloc[i, 1] :
#         inactive_freq.append(file.iloc[i, 2])
#
# print(sum(active_freq))
# print('\\\\')
# print(sum(inactive_freq))

