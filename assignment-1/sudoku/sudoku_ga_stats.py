from Sudoku import Sudoku
import matplotlib.pyplot as plt

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
rt , mem = [], []
lst = range(1000, 5001, 1000)
for population_size in lst:
    rti = obj.ga_runtime(n_iters=1000, population_size=population_size)
    memi = obj.ga_memory(n_iters=1000, population_size=population_size)

    rt.append(rti)
    mem.append(memi)

plt.plot(lst, rt)
plt.xlabel("Number of iterations")
plt.ylabel("Runtime")
plt.title("Runtime vs number of iterations")
plt.savefig("ga_runtime.png")

plt.plot(lst, mem)
plt.xlabel("Number of iterations")
plt.ylabel("Memory")
plt.title("Memory vs number of iterations")
plt.savefig("ga_memory.png")

plt.plot(lst, rt)
plt.xlabel("Population size")
plt.ylabel("Runtime")
plt.title("Runtime vs Population size")
plt.savefig("ga_runtime_vs_size.png")
plt.show()

plt.plot(lst, mem)
plt.xlabel("Population size")
plt.ylabel("Memory")
plt.title("Memory vs Population size")
plt.savefig("ga_memory_vs_size.png")