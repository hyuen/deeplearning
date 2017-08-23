f = open('submission.csv')

file = {}
for l in f:
    #print l
    ln = (l.split('.')[0].split('_')[1])
    file[ln] = l.strip()

for ln in sorted(file.keys()):
    print file[ln]
#img,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9
