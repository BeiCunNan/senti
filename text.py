from numpy import mean

l = '[SEP] please choose a correct sentiment category from { very negative, negative, neutral, positive, very positive }'
l = l.split(' ')
print(len(l))

l = [93.1, 55.3, 98.0, 78.4, 94.4, 87.5, 97.60, 89.60]
print(mean(l))
