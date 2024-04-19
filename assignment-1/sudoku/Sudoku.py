from numpy import array, uint8, zeros, copy
from numpy.random import shuffle, seed
from random import randint, choices, random, sample
from timeit import default_timer
import matplotlib.pyplot as plt
import tracemalloc

class DFS:
    def _bit_count(num: int) -> int:
        count = 0
        while num > 0:
            count += num & 1
            num >>= 1
        return count

    def _get_rightmost_bit_1_idx(num: int) -> int:
        if num:
            return DFS._bit_count((num & -num) - 1)
        return -1

    def __init__(self, lst: array) -> None:
        self._lst = lst.copy()
        
        self._zeros, self._rows, self._cols, self._blocks, self._valid = 0, {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0},\
        {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0},\
        {(0, 0): 0, (0, 1): 0, (0, 2): 0, (1, 0): 0, (1, 1): 0, (1, 2): 0, (2, 0): 0, (2, 1): 0, (2, 2): 0}, True

        for i in range(9):
            for j in range(9):
                val = int(self._lst[i, j])
                if val != 0:
                    if self._check(i, j, val):
                        self._rows[i] |= (1 << val)
                        self._cols[j] |= (1 << val)
                        self._blocks[i // 3, j // 3] |= (1 << val)
                    else:
                        self._valid = False
                else:
                    self._zeros |= (1 << i * 9 + j)
        
        self._val = 1

    def is_goal_state(self) -> bool:
        return self._zeros == 0 and self._valid

    def get_next_action(self) -> list[int] | None:
        if (idx := DFS._get_rightmost_bit_1_idx(self._zeros)) == -1:
            return None
        
        i, j = idx // 9, idx % 9
        while self._val < 10:
            if self._check(i, j, self._val):
                return [i, j, self._val]
            self._val += 1

        return None
        
    def result(self, action) -> None:
        i, j, val = action
        idx, bitmask = i * 9 + j, 1 << val

        self._rows[i] |= bitmask
        self._cols[j] |= bitmask
        self._blocks[i // 3, j // 3] |= bitmask
        self._zeros ^= (1 << idx)
        self._val = 1
        self._lst[i, j] = val

    def undo(self, action) -> None:
        i, j, val = action
        idx, bitmask = i * 9 + j, 1 << val

        self._rows[i] ^= bitmask
        self._cols[j] ^= bitmask
        self._blocks[i // 3, j // 3] ^= bitmask
        self._zeros |= (1 << idx)
        self._val = val + 1
        self._lst[i, j] = 0
        
    def _check(self, i: int, j: int, val: int) -> bool:
        return (self._rows[i] | self._cols[j] | self._blocks[i // 3, j // 3]) & (1 << val) == 0
    
    def _dfs(self, display: bool) -> bool:
        if display:
            print(self._lst)

        if self.is_goal_state():
            return True

        while (action := self.get_next_action()) != None:
            self.result(action)
            if self._dfs(display):
                return True
            self.undo(action)

        return False
    
    def solve(self, display: bool) -> tuple[bool, array]:
        success = self._dfs(display)
        return success, copy(self._lst)

class GA:
    MAX_FITNESS = 216

    def __init__(self, lst: list[int]) -> None:
        self._lst = lst
        self._blocks = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}
        
        for block_idx in range(9):
            for i in range(3):
                for j in range(3):
                    if (val := lst[block_idx // 3 * 3 + i, block_idx % 3 * 3 + j]) != 0:
                        self._blocks[block_idx] |= (1 << int(val))

    def _generate(self) -> str:
        s = ""
        blocks = self._blocks.copy()

        for block_idx in range(9):
            lst = []
            for val in range(1, 10):
                mask = 1 << val
                if blocks[block_idx] & mask == 0:
                    lst.append(val)

            for i in range(3):
                for j in range(3):
                    if (val := self._lst[block_idx // 3 * 3 + i, block_idx % 3 * 3 + j]) == 0:
                        val = choices(lst, k=1)[0]
                        lst.remove(val)

                    s += str(val)

        return s
    
    def _fitness(s: str) -> int:
        rows, cols, blocks = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0},\
        {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0},\
        {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}

        pos, fit = 0, GA.MAX_FITNESS
        for ch in s:
            val = ord(ch) - 48
            mask = 1 << val
            block_idx, block_offset = pos // 9, pos % 9
            r, c = block_idx // 3 * 3 + block_offset // 3, block_idx % 3 * 3 + block_offset % 3

            if (mask & rows[r]) != 0:
                fit -= 1
            else:
                rows[r] |= mask

            if (mask & cols[c]) != 0:
                fit -= 1
            else:
                cols[c] |= mask
            
            if (mask & blocks[block_idx]) != 0:
                fit -= 1
            else:
                blocks[block_idx] |= mask

            pos += 1

        return fit
    
    def _check_converge(lst: list[str]):
        max_fitness, min_fitness, best_str, worst_str, fit_lst = -1, GA.MAX_FITNESS, "", "", []

        for s in lst:
            fit = GA._fitness(s)
            fit_lst.append(fit)

            if fit > max_fitness:
                max_fitness, best_str = fit, s
            if fit < min_fitness:
                min_fitness, worst_str = fit, s

        return max_fitness == GA.MAX_FITNESS, max_fitness, min_fitness, best_str, worst_str, fit_lst
    
    def _crossover(s1: str, s2: str) -> str:
        idx, s = 0, ""
        for c1, c2 in zip(s1, s2):
            if idx % 9 == 0:
                p_idx = randint(0, 1)
            
            s += (c1 if p_idx == 0 else c2)
            
            idx += 1
        
        return s
    
    def _mutate(self, s: str) -> str:
        m = randint(0, 8) * 9
        lst = []
        for pos in range(m, m + 9):
            block_idx, block_offset = pos // 9, pos % 9
            r = block_idx // 3 * 3 + block_offset // 3
            c = block_idx % 3 * 3 + block_offset % 3
            if self._lst[r, c] == 0:
                lst.append(pos)

        if (len(lst) < 2):
            return s
        
        a, b = choices(lst, k=2)

        if a == b:
            return s

        pos, mutated_s = 0, ""
        for c in s:
            if pos == a:
                c = s[b]
            elif pos == b:
                c = s[a]
            
            mutated_s += c
            pos += 1
        
        return mutated_s

    def solve(self, n_iters: int, population_size: int, mutation_rate: float, display: bool):
        # check input
        if mutation_rate > 1:
            mutation_rate = 1
        elif mutation_rate < 0:
            mutation_rate = 0
        
        # if random_selection > 1:
        #     random_selection = 1
        # elif random_selection < 0:
        #     random_selection = 0
        # elif random_selection < .5:
        #     random_selection = .5

        if population_size <= 0:
            return False, self._s, []
        
        # initial population
        if display:
            print("Generating...")
        lst, population = [], [self._generate() for _ in range(population_size)]

        for iter in range(1, n_iters + 1):
            if display:
                print("Visiting iterate ", iter)
            converged, max_fitness, min_fitness, best_str, worst_str, fit_lst =  GA._check_converge(population)

            lst.append([max_fitness, min_fitness])

            if converged:
                print("Solution found at iterate", n_iters)
                return True, best_str, lst
            elif max_fitness == 0:
                return False, best_str, lst
            
            if display:
                print("   Best:", max_fitness, "-", best_str)
                print("  Worst:", min_fitness, "-", worst_str)

            # population = [x for _, x in sorted(zip(fit_lst, population))]

            # select parents
            # p = int(random_selection * population_size)
            # population = population[:p] + choices(population, k=population_size - p)
            population = choices(population, fit_lst, k=population_size)
            
            new_population = []
            for _ in range(population_size):
                s1, s2 = choices(population, k=2)
                child = GA._crossover(s1, s2)

                if random() < mutation_rate:
                    child = self._mutate(child)
                
                new_population.append(child)
                new_population.append(child)

            population = new_population

        return False, best_str, lst

class Sudoku:
    def Generator(k: int) -> array:
        if k < 0:
            k = 0
        elif k > 81:
            k = 81

        arr = zeros((9, 9), dtype=uint8)

        x1, x2, x3 = array(range(1, 10), dtype=uint8).reshape(3, 3), \
            array(range(1, 10), dtype=uint8).reshape(3, 3), \
                array(range(1, 10), dtype=uint8).reshape(3, 3)
        
        shuffle(x1)
        shuffle(x2)
        shuffle(x3)

        arr[0:3, 0:3] = x1
        arr[3:6, 3:6] = x2
        arr[6:9, 6:9] = x3

        obj = Sudoku(arr)
        _, arr = obj.DFS(display=False)

        arr = arr.flatten()
        arr[sample(range(0, 81), k=k)] = 0
        arr = arr.reshape(9, 9)

        return arr

    def _string_to_numpy(s: str) -> array:
        arr = zeros((9, 9), dtype=uint8)

        idx = 0
        for ch in s:
            block_idx, block_offset = idx // 9, idx % 9
            r, c = block_idx // 3 * 3 + block_offset // 3, block_idx % 3 * 3 + block_offset % 3
            arr[r][c] = ord(ch) - 48
            idx += 1
        
        return arr
    
    def __init__(self, lst: list[int]) -> None:
        self._lst = array(lst, dtype=uint8)

        if self._lst.shape != (9, 9):
            raise ValueError("List size must be 9x9")
        elif self._lst.max() > 9:
            raise ValueError("List must contain only value from 0 to 9")
        elif self._lst.min() < 0:
            raise ValueError("List must contain only value from 0 to 9")

    def DFS(self, display: bool = True) -> tuple[bool, array]:
        obj = DFS(self._lst)
        return obj.solve(display)

    def GA(self, n_iters, population_size: int = 1000, mutation_rate: float = .2, plot_fitness: bool = False, display: bool = False):
        obj = GA(self._lst)
        success, s, lst = obj.solve(n_iters, population_size, mutation_rate, display)
        print(Sudoku._string_to_numpy(s))
        if not success:
            print("Stuck at local maxima")
        # plot fitness vs iterate
        if plot_fitness:
            lst = array(lst)
            plt.plot(lst[:, 0], label="Best")
            plt.plot(lst[:, 1], label="Worst")
            plt.title("Fitness score vs. Iterate")
            plt.xlabel("Fitness score")
            plt.ylabel("Iterate")
            plt.legend()
            plt.show()

    def dfs_memory(self):
        print("Calculating memory...")
        tracemalloc.start()
        success, _ = self.DFS(display=False)
        m = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        print("Memory:", m)
        return m
    
    def dfs_runtime(self):
        print("Calculating runtime...")
        st = default_timer()
        self.DFS(display=False)
        et = default_timer()
        print("Runtime:", et - st)
        return et - st
    
    def ga_runtime(self, n_iters, population_size: int = 1000, mutation_rate: float = .2):
        print("Calculating runtime...")
        st = default_timer()
        self.GA(n_iters=n_iters, population_size=population_size, mutation_rate=mutation_rate)
        et = default_timer()
        print("Runtime:", et - st)
        return et - st
    
    def ga_memory(self, n_iters, population_size: int = 1000, mutation_rate: float = .2):
        print("Calculating memory...")
        tracemalloc.start()
        self.GA(n_iters=n_iters, population_size=population_size, mutation_rate=mutation_rate)
        m = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        print("Memory:", m)
        return m