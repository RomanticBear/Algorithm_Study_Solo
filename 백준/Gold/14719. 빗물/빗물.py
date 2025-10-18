H, W = map(int, input().split())
H_lst = list(map(int, input().split()))

ans = 0

# 왼쪽과 오른쪽에서 가장 높은 벽의 높이를 저장하는 리스트
left_max = [0] * W
right_max = [0] * W

# 왼쪽에서 가장 높은 벽 계산
left_max[0] = H_lst[0]
for i in range(1, W):
    left_max[i] = max(left_max[i - 1], H_lst[i])

# 오른쪽에서 가장 높은 벽 계산
right_max[W - 1] = H_lst[W - 1]
for i in range(W - 2, -1, -1):
    right_max[i] = max(right_max[i + 1], H_lst[i])

# 각 위치에서 빗물이 고일 수 있는 양 계산
for i in range(W):
    ans += min(left_max[i], right_max[i]) - H_lst[i]

print(ans)
