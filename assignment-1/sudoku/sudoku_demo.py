from Sudoku import Sudoku
from numpy.random import seed

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
obj.DFS()
obj.GA(n_iters=5000, display=True, plot_fitness=True)

# obj = Sudoku([[0] * 9] * 9)
# obj.dfs_runtime()