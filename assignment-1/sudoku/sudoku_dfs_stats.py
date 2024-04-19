from Sudoku import Sudoku
from pandas import DataFrame
import matplotlib.pyplot as plt

def dfs_memory() -> list:
    mem = []
    
    for k in range(82):
        while True:
            print(f"k = {k}")
            obj = Sudoku(Sudoku.Generator(k))
            print("Puzzle created successfully!")
            if (obj.dfs_runtime() < 1):
                mk = obj.dfs_memory()
                mem.append(mk)
                break
            print("Time limit exceeded")

    print("Saving...")
    DataFrame(mem, columns=["Peak memory"]).to_csv("dfs_peak_memory.csv")
    return mem

def dfs_runtime(lb: int = 0, ub: int = 82, group_size: int = 100, fn = "dfs_runtime.csv") -> list[list[int]]:
    if ub < lb:
        lb, ub = ub, lb
    if lb < 0:
        lb = 0
    if ub > 82:
        ub = 82

    runtime = []

    for k in range(lb, ub):
        rk = []
        for i in range(group_size):
            print(f"k = {k}, i = {i}")
            obj = Sudoku(Sudoku.Generator(k))
            print("Puzzle created successfully!")
            rki = obj.dfs_runtime()
            rk.append(rki)
        
        runtime.append(rk)

    print("Saving...")
    DataFrame(runtime).to_csv(fn)
    return runtime

mem = dfs_memory()
plt.plot(mem)
plt.title("Peak memory vs. Number of missing digits")
plt.ylabel("Peak memory")
plt.xlabel("Number of missing digits")
plt.savefig("dfs_peak_memory.png")
plt.show()

runtime = dfs_runtime(fn="df_runtime.csv")
runtime2 = dfs_runtime(lb=75, fn="df_runtime_2.csv")

mean_rt = DataFrame(runtime).mean(axis=1)
print(mean_rt)
plt.plot(mean_rt)
plt.title("Average runtime vs. Number of missing digits")
plt.xlabel("Number of missing digits")
plt.ylabel("Average runtime")
plt.savefig("Mean_Runtime.png")
plt.show()

mean_rt2 = DataFrame(runtime).mean(axis=1)
print(mean_rt2)
plt.plot(range(75, 82), mean_rt2)
plt.title("Average runtime vs. Number of missing digits (k = 75 to 81)")
plt.xlabel("Number of missing digits")
plt.ylabel("Average runtime")
plt.savefig("Mean_Runtime_2.png")
plt.show()