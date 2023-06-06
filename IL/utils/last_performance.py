folder = "2023-04-23_20H_32M"


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

p1 = accs1[-1].strip().split(',')
p2 = accs2[-1].strip().split(',')
p3 = accs3[-1].strip().split(',')
p4 = accs4[-1].strip().split(',')
p5 = accs5[-1].strip().split(',')

for i in range(len(p1)):
    acc1.append(float(p1[i]))
    acc2.append(float(p2[i]))
    acc3.append(float(p3[i]))
    acc4.append(float(p4[i]))
    acc5.append(float(p5[i]))

max = 0
max_sd = 0

mead_dice = (acc1[1] + acc2[1] + acc3[1] + acc4[1] + acc5[1]) / 5 * 100
mead_precision = (acc1[3] + acc2[3] + acc3[3] + acc4[3] + acc5[3]) / 5 * 100
mead_recall = (acc1[4] + acc2[4] + acc3[4] + acc4[4] + acc5[4]) / 5 * 100

sd = 0

sd += ((acc1[1]*100-mead_dice)*(acc1[1]*100-mead_dice))
sd += ((acc2[1]*100-mead_dice)*(acc2[1]*100-mead_dice))
sd += ((acc3[1]*100-mead_dice)*(acc3[1]*100-mead_dice))
sd += ((acc4[1]*100-mead_dice)*(acc4[1]*100-mead_dice))
sd += ((acc5[1]*100-mead_dice)*(acc5[1]*100-mead_dice))

sd /= 5

print(mead_dice)
print(sd)
print(mead_precision)
print(mead_recall)

acc1 = []
acc2 = []
acc3 = []
acc4 = []
acc5 = []

for i in range(len(accs1)):
    acc1.append(accs1[i].strip().split(','))
    acc2.append(accs2[i].strip().split(','))
    acc3.append(accs3[i].strip().split(','))
    acc4.append(accs4[i].strip().split(','))
    acc5.append(accs5[i].strip().split(','))

for i in range(len(acc1)):

    total = 0    
    total += float(acc1[i][1])
    total += float(acc2[i][1])
    total += float(acc3[i][1])
    total += float(acc4[i][1])
    total += float(acc5[i][1])

    total /= 5
    total *= 100

    sd = 0

    sd += ((float(acc1[i][1])*100-total)*(float(acc1[i][1])*100-total))
    sd += ((float(acc2[i][1])*100-total)*(float(acc2[i][1])*100-total))
    sd += ((float(acc3[i][1])*100-total)*(float(acc3[i][1])*100-total))
    sd += ((float(acc4[i][1])*100-total)*(float(acc4[i][1])*100-total))
    sd += ((float(acc5[i][1])*100-total)*(float(acc5[i][1])*100-total))

    sd /= 5

    if total >= max:
        max = total
        max_sd = sd

print("Best : ", max)
print(max_sd)