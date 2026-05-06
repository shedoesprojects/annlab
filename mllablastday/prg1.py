import csv

with open('mllablastday\Weather.csv') as f:
    data=list(csv.reader(f))
step=1
h=['0']*(len(data[0])-1)
for row in data[1:]:
    if row[-1]=='Yes':
        for i in range(len(h)):
            if h[i]=='0':
                h[i]=row[i]
            elif h[i]!=row[i]:
                h[i]='?'
        print(f"step {step} : hypothesis: ",h)
        step+=1
print("Most specific hypothesis : ",h)