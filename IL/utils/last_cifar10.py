folder0 = 'C:/Users/hamdo/Desktop/code/Influencer_learning/FL/Results/'
folder = "2023-08-26_9H"



f1 = open(folder0 + folder + '/participants1.txt', 'r')
f2 = open(folder0 + folder + '/participants2.txt', 'r')
f3 = open(folder0 + folder + '/participants3.txt', 'r')
f4 = open(folder0 + folder + '/participants4.txt', 'r')
f5 = open(folder0 + folder + '/participants5.txt', 'r')
f6 = open(folder0 + folder + '/participants6.txt', 'r')
f7 = open(folder0 + folder + '/participants7.txt', 'r')
f8 = open(folder0 + folder + '/participants8.txt', 'r')
f9 = open(folder0 + folder + '/participants9.txt', 'r')
f10 = open(folder0 + folder + '/participants10.txt', 'r')


accs1 = f1.readlines()
accs2 = f2.readlines()
accs3 = f3.readlines()
accs4 = f4.readlines()
accs5 = f5.readlines()
accs6 = f6.readlines()
accs7 = f7.readlines()
accs8 = f8.readlines()
accs9 = f9.readlines()
accs10 = f10.readlines()

acc1 = []
acc2 = []
acc3 = []
acc4 = []
acc5 = []
acc6 = []
acc7 = []
acc8 = []
acc9 = []
acc10 = []

for i in range(len(accs1)):
    acc1.append(float(accs1[i].strip()))
    acc2.append(float(accs2[i].strip()))
    acc3.append(float(accs3[i].strip()))
    acc4.append(float(accs4[i].strip()))
    acc5.append(float(accs5[i].strip()))
    acc6.append(float(accs6[i].strip()))
    acc7.append(float(accs7[i].strip()))
    acc8.append(float(accs8[i].strip()))
    acc9.append(float(accs9[i].strip()))
    acc10.append(float(accs10[i].strip()))

max = 0
max_sd = 0

for i in range(len(acc1)):

    total = 0    
    total += acc1[i]
    total += acc2[i]
    total += acc3[i]
    total += acc4[i]
    total += acc5[i]

    total /= 5
    total *= 100

    sd = 0

    sd += ((acc1[i]*100-total)*(acc1[i]*100-total))
    sd += ((acc2[i]*100-total)*(acc2[i]*100-total))
    sd += ((acc3[i]*100-total)*(acc3[i]*100-total))
    sd += ((acc4[i]*100-total)*(acc4[i]*100-total))
    sd += ((acc5[i]*100-total)*(acc5[i]*100-total))

    sd /= 5

    if total >= max:
        max = total
        max_sd = sd

total = 0    
total += acc1[-1]
total += acc2[-1]
total += acc3[-1]
total += acc4[-1]
total += acc5[-1]
total += acc6[-1]
total += acc7[-1]
total += acc8[-1]
total += acc9[-1]
total += acc10[-1]
# total *= 100
total /= 10

sd = 0

sd += ((acc1[-1]-total)*(acc1[-1]-total))
sd += ((acc2[-1]-total)*(acc2[-1]-total))
sd += ((acc3[-1]-total)*(acc3[-1]-total))
sd += ((acc4[-1]-total)*(acc4[-1]-total))
sd += ((acc5[-1]-total)*(acc5[-1]-total))
sd += ((acc6[-1]-total)*(acc6[-1]-total))
sd += ((acc7[-1]-total)*(acc7[-1]-total))
sd += ((acc8[-1]-total)*(acc8[-1]-total))
sd += ((acc9[-1]-total)*(acc9[-1]-total))
sd += ((acc10[-1]-total)*(acc10[-1]-total))

print("max: ", max)
print("max'ssd: ", max_sd)
print("final auc: ", total)
print("final sd: ", sd/10)