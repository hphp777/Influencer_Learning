folder = "2023-04-20_13H_18M"


f1 = open('C:/Users/hb/Desktop/code/Influencer_learning/IL/Results/' + folder + '/participants1.txt', 'r')
f2 = open('C:/Users/hb/Desktop/code/Influencer_learning/IL/Results/' + folder + '/participants2.txt', 'r')
f3 = open('C:/Users/hb/Desktop/code/Influencer_learning/IL/Results/' + folder + '/participants3.txt', 'r')
f4 = open('C:/Users/hb/Desktop/code/Influencer_learning/IL/Results/' + folder + '/participants4.txt', 'r')
f5 = open('C:/Users/hb/Desktop/code/Influencer_learning/IL/Results/' + folder + '/participants5.txt', 'r')


accs1 = f1.readlines()
accs2 = f2.readlines()
accs3 = f3.readlines()
accs4 = f4.readlines()
accs5 = f5.readlines()

acc1 = []
acc2 = []
acc3 = []
acc4 = []
acc5 = []

for i in range(len(accs1)):
    acc1.append(float(accs1[i].strip()))
    acc2.append(float(accs2[i].strip()))
    acc3.append(float(accs3[i].strip()))
    acc4.append(float(accs4[i].strip()))
    acc5.append(float(accs5[i].strip()))

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

print(max)
print(max_sd)