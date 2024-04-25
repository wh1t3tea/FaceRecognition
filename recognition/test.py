from callbacks import EarlyStop

test_a = list(range(0, 11))[::-1]
earlystop = EarlyStop("something")

for i in range(10):
    earlystop(test_a[i], i)
