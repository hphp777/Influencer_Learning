folder0 = 'C:/Users/hamdo/Desktop/code/Influencer_learning/IL/Results/'
folder = "2023-08-23_20H_28M"



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
f11 = open(folder0 + folder + '/participants11.txt', 'r')
f12 = open(folder0 + folder + '/participants12.txt', 'r')
f13 = open(folder0 + folder + '/participants13.txt', 'r')
f14 = open(folder0 + folder + '/participants14.txt', 'r')
f15 = open(folder0 + folder + '/participants15.txt', 'r')
f16 = open(folder0 + folder + '/participants16.txt', 'r')


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
accs11 = f11.readlines()
accs12 = f12.readlines()
accs13 = f13.readlines()
accs14 = f14.readlines()
accs15 = f15.readlines()
accs16 = f16.readlines()

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
acc11 = []
acc12 = []
acc13 = []
acc14 = []
acc15 = []
acc16 = []

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
    acc11.append(float(accs11[i].strip()))
    acc12.append(float(accs12[i].strip()))
    acc13.append(float(accs13[i].strip()))
    acc14.append(float(accs14[i].strip()))
    acc15.append(float(accs15[i].strip()))
    acc16.append(float(accs16[i].strip()))

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
total += acc11[-1]
total += acc12[-1]
total += acc13[-1]
total += acc14[-1]
total += acc15[-1]
total += acc16[-1]
# total *= 100
total /= 16

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
sd += ((acc11[-1]-total)*(acc11[-1]-total))
sd += ((acc12[-1]-total)*(acc12[-1]-total))
sd += ((acc13[-1]-total)*(acc13[-1]-total))
sd += ((acc14[-1]-total)*(acc14[-1]-total))
sd += ((acc15[-1]-total)*(acc15[-1]-total))
sd += ((acc16[-1]-total)*(acc16[-1]-total))

print("max: ", max)
print("max'ssd: ", max_sd)
print("final auc: ", total)
print("final sd: ", sd/16)