import time
import tracemalloc
from matplotlib import pyplot as plt
from minesweeper_core import MineSweeperCore

def sum_of_list(ls):
    sum = 0.0
    for i in ls:
        sum += i
    return sum

if __name__ == "__main__":
    i = 0
    trace_mem = []
    trace_ex_time = []
    x_axis = []
    while i <= 100:
        tracemalloc.start()
        x_axis.append(i)
        minesweeper = MineSweeperCore(10, 10, i)
        
        start_time = time.time()
        
        print("DFS Solution with i = {}:".format(i))
        if minesweeper.dfs_solution(False) and minesweeper.check_error_map() and minesweeper.check_completed_map():
            print("\nSuccessful solution!\n")
            # minesweeper.official_print_map()
            print("Thanks for playing!")
        else:
            print("ERROR!")
            exit(1)
        end_time = time.time()
        
        print("-" * 30)
        mem = round(tracemalloc.get_traced_memory()[1]/(1024), 2)
        tracemalloc.stop()
        print("The most memory usage: {} KB".format(mem))
        print("Excecuting time: {} s\n".format(end_time - start_time))
        trace_mem.append(mem)
        trace_ex_time.append(round((end_time - start_time)*1000, 2))
        i += 1
    print("===" * 30)
    print("Memory tracing: ", trace_mem)
    print("Executing time tracing: ", trace_ex_time)
    plt.plot(x_axis, trace_mem, label = "Memory (KB)")
    plt.plot(x_axis, trace_ex_time, label = "Time (ms)")
    plt.title("Memory usage and Executing time")
    plt.xlabel("Number of mines")
    plt.ylabel("")
    plt.legend()
    plt.grid(True)

    plt.show()