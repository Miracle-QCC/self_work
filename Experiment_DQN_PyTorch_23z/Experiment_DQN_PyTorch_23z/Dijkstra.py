def Dijkstra(network, s1, d1):
    path = []
    n1 = len(network)
    fmax = float('inf')
    w = [[0 for y1 in range(n1)] for j1 in range(n1)]
    book = [0 for y1 in range(n1)]
    dis = [fmax for i1 in range(n1)]
    book[s1] = 1
    midpath = [-1 for i1 in range(n1)]
    u1=-1
    for i1 in range(n1):
        for j1 in range(n1):
            if network[i1, j1] != 0:
                w[i1][j1] = network[i1, j1]  # 0→max
            else:
                w[i1][j1] = fmax
            if i1 == s1 and network[i1, j1] != 0:
                dis[j1] = network[i1, j1]
    for i1 in range(n1 - 1):
        min = fmax
        for j1 in range(n1):
            if book[j1] == 0 and dis[j1] < min:
                min = dis[j1]
                u1 = j1
        book[u1] = 1
        for v1 in range(n1):
            if dis[v1] > dis[u1] + w[u1][v1]:
                dis[v1] = dis[u1] + w[u1][v1]
                midpath[v1] = u1
    if dis[d1]!=fmax:
        j1 = d1
        path.append(d1)
        while (midpath[j1] != -1):
            path.append(midpath[j1])
            j1 = midpath[j1]
        path.append(s1)
        path.reverse()
        return path