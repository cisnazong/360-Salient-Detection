import sys
MAXL = 200000000000000
def solve(map,record, i, j, n ,m, p):
    if i == n:
        return map[i]

    if record[i,j] == 0:
        record[i,j] = 1
    if record[i,j] == 1:
        return MAXL
    else:
        p=[]
        length = 0
        if record[i-1,j] == 0:
            p.append(solve (map,record, i - 1, j, n ,m, p))
        if record[i - 1, j] == 0:
            p.append (solve (map, record, i - 1, j, n, m, p))
        if record[i + 1, j] == 0:
            p.append (solve (map, record, i - 1, j, n, m, p))
        if record[i, j-1] == 0:
            p.append (solve (map, record, i - 1, j, n, m, p))
        if record[i, j + 1] == 0:
            p.append (solve (map, record, i - 1, j, n, m, p))
        length = min(p)
        return length
if __name__ == "__main__":
    t = list(map(int, line.split()))
    n,m = t[0],t[1]
    map=[]
    record=[]
    for i in range(n):
        line = sys.stdin.readline().strip()
        values = list(map(int, line.split()))
        map.append(values)
        record.append([0 for j in range(m)])
    ans=[]
    solve(map,record,)

