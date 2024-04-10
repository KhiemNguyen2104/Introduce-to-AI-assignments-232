import random
import itertools
from copy import deepcopy

# ----------------------------------
# The value of a cell:
#      -3: masked
#      -2: empty
#      -1: bomb
#      0 - 8: number of bombs around
# ----------------------------------

class MineSweeperCore:
    def __init__(self, rows, columns, mines):
        self.rows = rows
        self.columns = columns
        self.mines = mines
        self.map = self.creat_map()
        self.place_mines()
        self.mines_position = self.get_mines_postion()
        self.place_numbers()
        self.official_map = self.minesweeper_map()

    def in_range(self, x, y):
        return x >= 0 and x < self.rows and y >= 0 and y < self.columns

    def get_mines_postion(self):
        list_of_mines = []
        for x in range(self.rows):
            for y in range(self.columns):
                if self.map[x][y] == -1:
                    list_of_mines.append((x, y))
        return list_of_mines
    
    def num_of_mines(self):
        num_of_mines = 0
        for row in self.map:
            for cell in row:
                if cell == -1:
                    num_of_mines += 1
        return num_of_mines

    def creat_map(self):
        MSMap = [[-2]*self.columns for _ in range(self.rows)]
        return MSMap
        
    def place_mines(self):
        cells = [(x, y) for x in range(self.rows) for y in range(self.columns)]
        bomb_cells = random.sample(cells, self.mines)
        for x, y in bomb_cells:
            self.map[y][x] = -1
    
    def get_bomb_around(self, x, y):
        num_of_bombs = 0
        for l in range(-1, 2):
            for o in range(-1, 2):
                if (l != 0 or o != 0) and self.in_range(x + l, y + o) and self.map[x + l][y + o] == -1:
                    num_of_bombs += 1
        return num_of_bombs

    def get_empty_around(self, x, y):
        num_of_empty = 0
        for l in range(-1, 2):
            for o in range(-1, 2):
                if (l != 0 or o != 0) and self.in_range(x + l, y + o) and self.map[x + l][y + o] == -2:
                    num_of_empty += 1
        return num_of_empty

    def check_completed(self, x, y):
        if self.map[x][y] == -2:
            checked = False
            for l in range(-1, 2):
                for o in range(-1, 2):
                    if (l != 0 or o != 0) and self.in_range(x + l, y + o) and self.map[x + l][y + o] != -1 and self.map[x + l][y + o] != -2:
                        checked = self.check_completed(x + l, y + o)
            return checked
        elif self.map[x][y] != -1:
            val = self.map[x][y]
            for l in range(-1, 2):
                for o in range(-1, 2):
                    if (l != 0 or o != 0) and self.in_range(x + l, y + o) and self.map[x + l][y + o] == -1:
                        val -= 1
            return (val == 0)
        else:
            return True

    def place_numbers(self):
        for cell in self.get_mines_postion():
            nums = random.choice([1, 2])
            for _ in range(nums):
                while (1):
                    l = random.choice([-1, 0, 1])
                    o = random.choice([-1, 0, 1])
                    if (l != 0 or o != 0):
                        if self.in_range(cell[0] + l, cell[1] + o) and self.map[cell[0] + l][cell[1] + o] == -2:
                            self.map[cell[0] + l][cell[1] + o] = self.get_bomb_around(cell[0] + l, cell[1] + o)
                            break
                    else:
                        continue
        for x in range(self.rows):
            for y in range(self.columns):
                if not self.check_completed(x, y) and self.map[x][y] == -2:
                    choice = random.choice(range(1, 101))
                    if choice >= 30 and self.get_empty_around(x, y) > 0:
                        ls = []
                        for l in range(-1, 2):
                            for o in range(-1, 2):
                                if (l != 0 or o != 0) and self.in_range(x + l, y + o) and self.map[x + l][y + o] == -2:
                                    ls.append((x + l, y + o))
                        position = random.choice(ls)
                        self.map[position[0]][position[1]] = self.get_bomb_around(position[0], position[1])
                    else:
                        self.map[x][y] = self.get_bomb_around(x, y)

    def print_map(self):
        max_row_length = max(len(' | '.join(map(str, row))) for row in self.map)
        for i, row in enumerate(self.map):
            print(' | '.join(map(str, row)))
            if i < len(self.map) - 1:
                print('-' * max_row_length)

    # Official functions

    def Heuristic_function(self):
        pass

    def official_get_number_positions(self):
        ls = []
        for x in range(self.rows):
            for y in range(self.columns):
                if self.official_map[x][y] >= 0:
                    ls.append((x, y))
        return ls      

    def official_check_completed(self, x, y):
        if self.official_map[x][y] >= 0:
            val = self.official_map[x][y]
            cells = len(self.official_get_cells_around(x, y))
            return (val == len(self.official_get_bombs_around(x, y)) and (cells - val) == len(self.official_get_emptys_around(x, y)) + len(self.official_get_numbers_around(x, y)))
        else:
            return True

    def official_print_map(self):
        max_row_length = max(len(' | '.join(map(str, row))) for row in self.official_map)
        for i, row in enumerate(self.official_map):
            print(' | '.join(map(str, row)))
            if i < len(self.official_map) - 1:
                print('-' * max_row_length)

    def minesweeper_map(self):
        official_map = deepcopy(self.map)
        for x in range(self.rows):
            for y in range(self.columns):
                if official_map[x][y] == -1 or official_map[x][y] == -2:
                    official_map[x][y] = -3
        return official_map

    def move(self, x, y, action):
        if action == 'l':
            self.official_map[x][y] = -2
        elif action == 'r':
            self.official_map[x][y] = -1
    
    def official_get_bombs_around(self, x, y):
        ls = []
        for l in range(-1, 2):
            for o in range(-1, 2):
                if (l != 0 or o != 0) and self.in_range(x + l, y + o) and self.official_map[x + l][y + o] == -1:
                    ls.append((x + l, y + o))
        return ls
    
    def official_get_cells_around(self, x, y):
        ls = []
        for l in range(-1, 2):
            for o in range(-1, 2):
                if (l != 0 or o != 0) and self.in_range(x + l, y + o):
                    ls.append((x + l, y + o))
        return ls

    def official_get_numbers_around(self, x, y):
        ls = []
        for l in range(-1, 2):
            for o in range(-1, 2):
                if (l != 0 or o != 0) and self.in_range(x + l, y + o) and self.official_map[x + l][y + o] >= 0:
                    ls.append((x + l, y + o))
        return ls

    def official_get_emptys_around(self, x, y):
        ls = []
        for l in range(-1, 2):
            for o in range(-1, 2):
                if (l != 0 or o != 0) and self.in_range(x + l, y + o) and self.official_map[x + l][y + o] == -2:
                    ls.append((x + l, y + o))
        return ls
    
    def official_get_masked_around(self, x, y):
        ls = []
        for l in range(-1, 2):
            for o in range(-1, 2):
                if (l != 0 or o != 0) and self.in_range(x + l, y + o) and self.official_map[x + l][y + o] == -3:
                    ls.append((x + l, y + o))
        return ls

    def official_get_error(self, x, y):
        if self.official_map[x][y] >= 0 and self.official_map[x][y] < len(self.official_get_bombs_around(x, y)):
            return False
        if self.official_map[x][y] >= 0 and len(self.official_get_bombs_around(x, y)) + len(self.official_get_masked_around(x, y)) < self.official_map[x][y]:
            return False
        return True

    def const_rules(self, x, y, stack):
        if self.official_map[x][y] >= 0: 
            bombs_around = len(self.official_get_bombs_around(x, y))
            masked_around = len(self.official_get_masked_around(x, y))
            
            if self.official_map[x][y] - bombs_around == masked_around:
                for l in range(-1, 2):
                    for o in range(-1, 2):
                        if (l != 0 or o != 0) and self.in_range(x + l, y + o) and self.official_map[x + l][y + o] == -3:
                            self.move(x + l, y + o, 'r')
                            stack.append((x + l, y + o))
            elif self.official_map[x][y] - bombs_around == 0:
                for l in range(-1, 2):
                    for o in range(-1, 2):
                        if (l != 0 or o != 0) and self.in_range(x + l, y + o) and self.official_map[x + l][y + o] == -3:
                            self.move(x + l, y + o, 'l')
                            stack.append((x + l, y + o))
            elif not self.official_get_error(x, y):
                return False
        elif self.official_map[x][y] == -2 or self.official_map[x][y] == -1:
            for pos in self.official_get_numbers_around(x, y):
                # if not self.official_check_completed(pos[0], pos[1]):
                stack.append(pos)
        return True

    def dfs_const_rules(self, x, y):
        stack = [(x, y)]
        while len(stack) > 0:
            (a, b) = stack.pop()
            self.const_rules(a, b, stack)
        if not self.check_completed_map():
            return False
        return True
            
    def check_completed_map(self):
        for pos in self.official_get_number_positions():
            if not self.official_check_completed(pos[0], pos[1]):
                return False

    def check_error_map(self):
        for pos in self.official_get_number_positions():
            if not self.official_get_error(pos[0], pos[1]):
                return False
        return True   

    def dfs_solution(self):
        if self.check_completed_map():
            return True
        else:
            for x in range(self.rows):
                for y in range(self.columns):
                    self.dfs_const_rules(x, y)
            if not self.check_error_map():
                return False
            case_map = deepcopy(self.official_map)
            print("The current map:\n")
            self.official_print_map()
            print("The back-up map: {}\n".format(case_map))
            ls = self.official_get_number_positions()
            max_value = 0
            max_pos = (0, 0)
            for pos in ls:
                if not self.official_check_completed(pos[0], pos[1]):
                    remainder_bombs = self.official_map[pos[0]][pos[1]] - len(self.official_get_bombs_around(pos[0], pos[1]))
                    remainder_maskeds = self.official_get_masked_around(pos[0], pos[1])
                    print("Details about {}: {}\n  - Remainder bombs: {}\n  - Remainder maskeds: {}\n".format(pos, self.official_map[pos[0]][pos[1]], remainder_bombs, remainder_maskeds))
                    if max_value < (remainder_bombs/len(remainder_maskeds)):
                        max_value = self.official_map[pos[0]][pos[1]]
                        max_pos = (pos[0], pos[1])
                    elif max_value == (remainder_bombs/len(remainder_maskeds)):
                        if self.official_map[pos[0]][pos[1]] < self.official_map[max_pos[0]][max_pos[1]]:
                            max_pos = (pos[0], pos[1])
            remainder_bombs = self.official_map[max_pos[0]][max_pos[1]] - len(self.official_get_bombs_around(max_pos[0], max_pos[1]))
            maskeds = self.official_get_masked_around(max_pos[0], max_pos[1])
            masked_combinations = list(itertools.combinations(maskeds, remainder_bombs))
            print("The position for go deeper is: {} with {} bombs remain".format(max_pos, remainder_bombs))
            print("Position of maskeds: {}".format(maskeds))
            print("The combinations: {}\n".format(masked_combinations))
            for pos in masked_combinations:
                temp_backup_map = deepcopy(self.official_map)
                for e in pos:
                    self.move(e[0], e[1], 'r')
                self.dfs_const_rules(max_pos[0], max_pos[1])
                print("Map in checking:\n")
                self.official_print_map()
                print("\n")
                if not self.check_error_map():
                    print("Return the backup map A: {}\n".format(temp_backup_map))
                    self.official_map = deepcopy(temp_backup_map)
                else:
                    if not self.dfs_solution():
                        print("Return the backup map B: {}\n".format(temp_backup_map))
                        self.official_map = deepcopy(temp_backup_map)
                    else:
                        return True
            if not self.check_error_map() or not self.check_completed_map():
                return False
        return True

    def hill_climbing_solution(self):
        if self.check_completed_map():
            return True
        else:
            for x in range(self.rows):
                for y in range(self.columns):
                    self.dfs_const_rules(x, y)
            if not self.check_error_map():
                return False
            case_map = deepcopy(self.official_map)
            print("The current map:\n")
            self.official_print_map()
            print("The back-up map: {}\n".format(case_map))
            ls = self.official_get_number_positions()
            max_value = 0
            max_pos = (0, 0)
            for pos in ls:
                if not self.official_check_completed(pos[0], pos[1]):
                    remainder_bombs = self.official_map[pos[0]][pos[1]] - len(self.official_get_bombs_around(pos[0], pos[1]))
                    remainder_maskeds = self.official_get_masked_around(pos[0], pos[1])
                    print("Details about {}: {}\n  - Remainder bombs: {}\n  - Remainder maskeds: {}\n".format(pos, self.official_map[pos[0]][pos[1]], remainder_bombs, remainder_maskeds))
                    if max_value < (remainder_bombs/len(remainder_maskeds)):
                        max_value = self.official_map[pos[0]][pos[1]]
                        max_pos = (pos[0], pos[1])
                    elif max_value == (remainder_bombs/len(remainder_maskeds)):
                        if self.official_map[pos[0]][pos[1]] < self.official_map[max_pos[0]][max_pos[1]]:
                            max_pos = (pos[0], pos[1])
            remainder_bombs = self.official_map[max_pos[0]][max_pos[1]] - len(self.official_get_bombs_around(max_pos[0], max_pos[1]))
            maskeds = self.official_get_masked_around(max_pos[0], max_pos[1])
            masked_combinations = list(itertools.combinations(maskeds, remainder_bombs))
            print("The position for go deeper is: {} with {} bombs remain".format(max_pos, remainder_bombs))
            print("Position of maskeds: {}".format(maskeds))
            print("The combinations: {}\n".format(masked_combinations))
            for pos in masked_combinations:
                temp_backup_map = deepcopy(self.official_map)
                for e in pos:
                    self.move(e[0], e[1], 'r')
                self.dfs_const_rules(max_pos[0], max_pos[1])
                print("Map in checking:\n")
                self.official_print_map()
                print("\n")
                if not self.check_error_map():
                    print("Return the backup map A: {}\n".format(temp_backup_map))
                    self.official_map = deepcopy(temp_backup_map)
                else:
                    if not self.dfs_solution():
                        print("Return the backup map B: {}\n".format(temp_backup_map))
                        self.official_map = deepcopy(temp_backup_map)
                    else:
                        return True
            if not self.check_error_map() or not self.check_completed_map():
                return False
        return True

if __name__ == "__main__":
    minesweeper = MineSweeperCore(15, 15, 14)
    minesweeper.print_map()
    # print(minesweeper.official_get_masked_around(0, 1))
    if minesweeper.dfs_solution() and not minesweeper.check_error_map() and minesweeper.check_completed_map():
        print("Successful solution!\n")
    minesweeper.official_print_map()
    print("Thanks for playing!")
