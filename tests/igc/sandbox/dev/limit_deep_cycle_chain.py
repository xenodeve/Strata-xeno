head = []
cur = head
for i in range(50000):
    nxt = []
    cur.append(nxt)
    cur = nxt
cur.append(head)
print('cycle built')
