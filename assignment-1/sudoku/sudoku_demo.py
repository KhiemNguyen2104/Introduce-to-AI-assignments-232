from Sudoku import Sudoku

lst = [
    [3,0,6,5,0,8,4,0,0],
    [5,2,0,0,0,0,0,0,0],
    [0,8,7,0,0,0,0,3,1],
    [0,0,3,0,1,0,0,8,0],
    [9,0,0,8,6,3,0,0,5],
    [0,5,0,0,9,0,6,0,0],
    [1,3,0,0,0,0,2,5,0],
    [0,0,0,0,0,0,0,7,4],
    [0,0,5,2,0,6,3,0,0]
]

obj = Sudoku(lst)
# obj.DFS()
# obj.GA(n_iters=1000, mutation_rate=.5)
print(obj.dfs_statistics())

# lst = [[9] * 9] * 9
# lst[0][0] = 0
# print(lst)
# obj = Sudoku(lst)
# obj.GA(n_iters=10)