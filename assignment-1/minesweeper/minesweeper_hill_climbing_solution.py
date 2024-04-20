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
        x_axis.append(i)
        minesweeper = MineSweeperCore(10, 10, i)

        t_array = []
        m_array = []
        print("Hill climbing Solution with i = {}:".format(i))
        for _ in range(5):
            tracemalloc.start()
            start_time = time.time()
            
            if minesweeper.hill_climbing_solution(False):
                pass
            else:
                print("ERROR!")
                exit(1)

            end_time = time.time()
            element_mem = round(tracemalloc.get_traced_memory()[1]/(1024), 2)
            tracemalloc.stop()
            t_array.append(end_time - start_time)
            m_array.append(element_mem)
            minesweeper = MineSweeperCore(10, 10, i)
        
        # print("Time array: ", t_array)
        # print("Memory array: ", m_array)
        print("\nSuccessful solution!\n")
        print("-" * 30)
        mem = sum_of_list(m_array)/5
        ex_time = sum_of_list(t_array)/5
        print("The most memory usage: {} KB".format(round(mem, 2)))
        print("Excecuting time: {} s\n".format(round(ex_time, 2)))
        trace_mem.append(round(mem, 2))
        trace_ex_time.append(round((ex_time)*1000, 2))
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

    # -----------------------------------------
    # Checking step by step

    # minesweeper = MineSweeperCore(10, 10, 50)
    
    # tracemalloc.start()
    # start_time = time.time()
        
    # # print("Hill climbing Solution with i = {}:".format(i))
    # if minesweeper.hill_climbing_solution(True):
    #     print("\nSuccessful solution!\n")
    #     minesweeper.official_print_map()
    # else:
    #     print("ERROR!")
    #     exit(1)
    # end_time = time.time()
        
    # print("-" * 30)
    # mem = round(tracemalloc.get_traced_memory()[1]/(1024), 2)
    # tracemalloc.stop()
    # print("The most memory usage: {} KB".format(mem))
    # print("Excecuting time: {} s\n".format(end_time - start_time))