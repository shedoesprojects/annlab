import csv

with open("mllablastday\Weather.csv") as f:
    data=list(csv.reader(f))
attr=len(data[0])-1
G=['?']*attr
S=['0']*attr
step=1
for row in data[1:]:
    if row[-1]=='Yes':
        for i in range(attr):
            if S[i]=='0':
                S[i]=row[i]
            elif S[i]!=row[i]:
                S[i]='?'
        print(f"Step {step} (+) S={S} G={G}")
    else:
        for i in range(attr):
            if S[i]!=row[i]:
                G[i]=S[i]
            else:
                G[i]='?'
        print(f"Step {step} (-) S={S} G={G}")
    step+=1
print("Most specific : ", S)
print("Most general : ", G)