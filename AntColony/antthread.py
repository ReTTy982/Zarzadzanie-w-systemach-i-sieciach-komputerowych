from __future__ import annotations
import math
from pathlib import Path
import random
import time
import re
import csv
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed

# When denominator's precision will bug out to be equal zero.
SMALL_FLOAT = 2.2250738585072014e-308


class AntColony:

    def __init__(self, alpha, beta, evaporation_rate, matrix, qas_value, ants_number):
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.matrix = matrix
        self.qas_value = qas_value

        self.pheromone_matrix = []
        self.ants_number = ants_number
        self.N = len(matrix)
        self.colony = []
        self.min_cost = 99999999999999999999
        self.best_path = []

    def init_pheromone(self):
        tau_zero = self.ants_number / self.init_distance()

        for i in range(self.N):
            self.pheromone_matrix.append([])
        for i in range(self.N):
            for j in range(self.N):
                self.pheromone_matrix[i].append(tau_zero)

    def init_distance(self):
        vertexes = list(range(0, self.N))
        random.shuffle(vertexes)
        cost = self.calculate_path_cost(vertexes)
        return cost

    def calculate_path_cost(self, path):
        cost = 0
        for i in range(self.N - 1):
            cost += self.matrix[path[i]][path[i + 1]]
        cost += self.matrix[path[-1]][path[0]]
        return cost

    def calculate_denominator(self, ant: Ant):
        denominator = 0.0
        atractivness = dict()
        for i in ant.possible_moves:
            pheromone_ammount = float(self.pheromone_matrix[ant.last_visited][i])
            distance = float(self.matrix[ant.last_visited][i])
            if distance == 0:
                distance = 0.00001
            atractivness[i] = pow(pheromone_ammount, self.alpha) * pow(1.0 / distance, self.beta)
            denominator += atractivness[i]

        return denominator, atractivness

    def change_pheromone(self):
        for i in range(self.N):
            for j in range(self.N):
                if i == j:
                    pass
                else:
                    self.pheromone_matrix[i][j] = self.pheromone_matrix[i][j] * self.evaporation_rate

        for ant in self.colony:
            cost = self.calculate_path_cost(ant.tabu)
            if cost < self.min_cost:
                self.min_cost = cost
                self.best_path = ant.tabu.copy()

            for i in range(self.N):
                if i == self.N - 1:
                    a = 0
                else:
                    self.pheromone_matrix[ant.tabu[i]][ant.tabu[i + 1]] += float(self.qas_value / cost)
            ant.reset()

    def pick_vertex(self, ant: Ant, denominator: float, atractivness: dict):
        sum = 0
        chance = random.random()
        sorted_list = sorted(atractivness.items(), key=lambda x: x[1])
        for struct in sorted_list:
            i = struct[0]
            if denominator == 0:
                denominator = SMALL_FLOAT
            sum += atractivness[i] / denominator
            if sum > chance:
                ant.pick_vertex(i)
                return

        vertex = sorted_list[-1][0]
        ant.pick_vertex(vertex)

    def run_algorithm(self, iter):
        self.init_pheromone()
        vertexes = list(range(0, self.N))
        starting_vertex = 0

        for i in range(self.ants_number):
            starting_vertex = random.choice(vertexes)
            self.colony.append(self.Ant(starting_vertex, i))

        for iterations in range(iter):
            for ant_index in range(len(self.colony)):
                ant = self.colony[ant_index]
                for move in range(self.N - 1):
                    if ant.first_move:
                        ant.possible_moves = list(set(vertexes.copy()) - set(ant.tabu))
                        ant.first_move = False

                    denominator, atractivness = self.calculate_denominator(ant)
                    self.pick_vertex(ant, denominator, atractivness)

            self.change_pheromone()
        return self.min_cost

    class Ant():
        def __init__(self, starting_vertex, id):
            self.tabu = [starting_vertex]
            self.possible_moves = None
            self.first_move = True
            self.last_visited = starting_vertex
            self.id = id

        def reset(self):
            self.tabu = self.tabu[:1]
            self.first_move = True
            self.last_visited = self.tabu[0]

        def pick_vertex(self, vertex):
            self.tabu.append(vertex)
            self.possible_moves.remove(vertex)
            self.last_visited = vertex


def better_config(file):
    folder = Path('Dane')
    file = folder / file
    with open(f"{file}", 'r') as f:
        t = int(f.readline().strip())
        l = [[] for _ in range(t)]
        row = 0
        column = 0
        liczba = ""
        read = f.read()
        read = ' '.join(read.split())
        for i in read:
            if i == " " or i == "\n":
                l[row].append(int(liczba))
                liczba = ""
                column += 1
                if column == t:
                    column = 0
                    row += 1
            else:
                liczba += i
        l[row].append(int(liczba))
        return l


def get_ini():
    tsp = {}
    with open("config.ini", 'r') as f:
        files_nr = int(f.readline().strip())
        for i in range(files_nr):
            x = f.readline().strip().split(" ")
            tsp[x[0]] = x[1:8]

        content = f.read()
        output = re.findall(r'#\w+', content)
        output = output[0]
        output = output[1:] + ".csv"

    return tsp, output


def benchmark(colony, iterations):
    start_time = time.time()
    end_cost = colony.run_algorithm(iterations)
    end_time = time.time()
    run_time = end_time - start_time
    print(f"{start_time} , {end_time}, {end_time - start_time}")
    writer.writerow([colony.N,run_time,end_cost,colony.best_path])

    return run_time, end_cost, colony.best_path


def rotate(lst):
    for i in range(len(lst)):
        if lst[i] == 0:
            lst = lst[i:] + lst[:i]
    return lst


def run_algorithm_and_write(args):
    file_name, alpha, beta, evaporation_rate, matrix, qvalue, ants_number, iterations, opt_cost, tries = args
    colony = AntColony(alpha, beta, evaporation_rate, matrix, qvalue, ants_number)
    suma = 0
    for i in range(tries):
        run_time, end_cost, path = benchmark(colony, iterations)
        suma += end_cost
        results.append([file_name, run_time, end_cost, rotate(path)])
    print(f"{file_name}: {round((((suma/tries) - opt_cost)/opt_cost)*100,2)}")
    return run_time

results = []

if __name__ == '__main__':
    start_time_thread = time.time()
    alpha = 1.0
    beta = 3.0
    evaporation_rate = 0.5
    qvalue = 100.0
    files, output = get_ini()
    f = open(output, 'w', newline='')
    writer = csv.writer(f, delimiter=";")
    writer.writerow(["Plik", "Czas[s]", "Koszt", "Sciezka"])

    results = []

    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(
                run_algorithm_and_write,
                (
                    file_name,
                    float(files[file_name][3]),
                    float(files[file_name][4]),
                    float(files[file_name][5]),
                    better_config(file_name),
                    float(files[file_name][6]),
                    len(better_config(file_name)),  # Poprawione
                    int(files[file_name][2]),
                    float(files[file_name][0]),
                    int(files[file_name][1])
                )
            )
            for file_name in files.keys()
        ]

        total_time = 0
        for future in as_completed(futures):
            total_time += future.result()

    print(f"Koniec: {total_time}")
    print(f"Koniec dobry {time.time() - start_time_thread}")
