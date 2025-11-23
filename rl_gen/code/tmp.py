import sys
from collections import deque

def main():
    t = int(sys.stdin.readline())
    for _ in range(t):
        n = int(sys.stdin.readline())
        adj = [[] for _ in range(n + 1)]
        for _ in range(n - 1):
            u, v = map(int, sys.stdin.readline().split())
            adj[u].append(v)
            adj[v].append(u)
        
        parent = [0] * (n + 1)
        children_count = [0] * (n + 1)
        q = deque()
        q.append(1)
        parent[1] = -1
        
        while q:
            u = q.popleft()
            for v in adj[u]:
                if parent[v] == 0 and v != parent[u]:
                    parent[v] = u
                    children_count[v] = 1
                    q.append(v)
        
        total = 0
        for u in range(1, n + 1):
            c = children_count[u]
            total += (c + 1) // 2
        
        print(total)

if __name__ == "__main__":
    main()