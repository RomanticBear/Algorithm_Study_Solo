from collections import deque

N, M = map(int, input().split())
arr = [list(map(int, input().split())) for _ in range(N)]

# 빈 공간 리스트와 바이러스 리스트
empty_spaces = [(i, j) for i in range(N) for j in range(M) if arr[i][j] == 0]
viruses = [(i, j) for i in range(N) for j in range(M) if arr[i][j] == 2]

# BFS 함수
def bfs(walls):
    temp_arr = [row[:] for row in arr]  # 원본 배열 복사
    for i, j in walls:
        temp_arr[i][j] = 1

    q = deque(viruses)
    visited = [[False] * M for _ in range(N)]

    for i, j in viruses:
        visited[i][j] = True

    while q:
        ci, cj = q.popleft()

        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = ci + di, cj + dj
            if 0 <= ni < N and 0 <= nj < M and not visited[ni][nj] and temp_arr[ni][nj] == 0:
                visited[ni][nj] = True
                temp_arr[ni][nj] = 2
                q.append((ni, nj))

    return sum(row.count(0) for row in temp_arr)

# 백트래킹 with BFS
def dfs(wall_count, start):
    global max_safe_area

    if wall_count == 3:
        safe_area = bfs(selected_walls)
        if safe_area > max_safe_area:
            max_safe_area = safe_area
        return

    for idx in range(start, len(empty_spaces)):
        selected_walls.append(empty_spaces[idx])
        dfs(wall_count + 1, idx + 1)
        selected_walls.pop()

selected_walls = []
max_safe_area = 0
dfs(0, 0)
print(max_safe_area)