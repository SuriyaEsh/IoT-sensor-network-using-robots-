import numpy as np
from scipy.spatial.distance import cdist
import networkx as nx
import sys
import random

def Greedy1_algorithm(Adj_matrix, prizes, budget):

    number_nodes = Adj_matrix.shape[0]
    is_visited = np.zeros(number_nodes, dtype=bool)
    curr_node = 0
    is_visited[curr_node] = True
    path = [curr_node]
    total_cost = 0
    total_prizes = prizes[curr_node]
    remaining_energy = budget
    while True:
        if all(is_visited):
            break
        distances_to_unvisited = []
        for i in range(number_nodes):
            if is_visited[i]:
                distances_to_unvisited.append(np.inf)
            else:
                distances_to_unvisited.append(Adj_matrix[curr_node][i])
        max_ratio = -np.inf
        next_node = -1
        for node_index in range(len(is_visited)):
            if not is_visited[node_index]:
                ratio = prizes[node_index] / distances_to_unvisited[node_index]
                if ratio > max_ratio:
                    max_ratio = ratio
                    next_node = node_index
        distance_to_next_node = distances_to_unvisited[next_node]
        if distance_to_next_node > remaining_energy:
            break
        is_visited[next_node] = True
        path.append(next_node)
        total_cost += distance_to_next_node
        total_prizes += prizes[next_node]
        remaining_energy -= distance_to_next_node
        curr_node = next_node
    path.append(0)
    total_cost =total_cost+ Adj_matrix[curr_node][0]
    return (path, total_cost, total_prizes, remaining_energy)


def Greedy2_algorithm(Adj_matrix, prizes, budget):
    n = len(Adj_matrix)
    is_visited = set([0])
    remaining_energy = budget
    total_distance = 0
    total_prizes = prizes[0]
    curr_node = 0
    path = [curr_node]
    while len(is_visited) < n:
        max_ratio = float('-inf')
        next_node = None
        for i in [x for x in range(n) if x not in is_visited]:
            distance = Adj_matrix[curr_node][i]
            if distance <= remaining_energy:
                ratio = prizes[curr_node] / distance
                if ratio > max_ratio:
                    max_ratio = ratio
                    next_node = i
        if next_node is None:
            break
        is_visited.add(next_node)
        remaining_energy -= Adj_matrix[curr_node][next_node]
        total_distance += Adj_matrix[curr_node][next_node]
        total_prizes += prizes[next_node]
        curr_node = next_node
        path.append(curr_node)
    total_distance += Adj_matrix[curr_node][0]
    path.append(0)
    return path, total_distance, total_prizes, remaining_energy


def MARL_algorithm(Adj_matrix, prizes, budget, num_agents=100, max_iterations=5000):
    n = len(Adj_matrix)
    A_matrix = np.zeros((n, n))
    R_matrix = np.zeros((n, n))
    P_vector = prizes.copy()
    E_budget = budget
    best_path = []
    best_cost = float('inf')
    best_prize = 0
    remaining_energy = E_budget
    for episode in range(max_iterations):
        V_vector = random.sample(range(n), num_agents)
        for j in range(num_agents):
            u_node = V_vector[j]
            Q_values = [A_matrix[u_node][w] + (P_vector[w] / (1 + R_matrix[u_node][w])) for w in range(n)]
            w_star_node = np.argmax(Q_values)
            for i in range(n):
                R_matrix[i][w_star_node] += Adj_matrix[i][w_star_node]
            Q_star_value = np.max(Q_values)
            A_matrix[u_node][w_star_node] = A_matrix[u_node][w_star_node] + (1 / (w_star_node + 1)) * (R_matrix[u_node][w_star_node] + np.sum(A_matrix, axis=1).max() - A_matrix[u_node][w_star_node])
            remaining_energy -= Adj_matrix[u_node][w_star_node]
            prize_value = P_vector[w_star_node]
            P_vector[w_star_node] = 0
        path_array = np.zeros(num_agents, dtype=int)
        cost_value = 0
        total_prize_value = 0
        for j in range(num_agents):
            u_node = V_vector[j]
            path_array[j] = u_node
            if j == 0:
                cost_value += Adj_matrix[u_node][w_star_node]
            else:
                cost_value += Adj_matrix[V_vector[j - 1]][u_node]
            total_prize_value += prizes[u_node] - P_vector[u_node]
        if total_prize_value - cost_value > best_prize - best_cost:
            best_path = path_array
            best_cost = cost_value
            best_prize = total_prize_value
            remaining_energy = budget - cost_value
            P_vector = prizes.copy()
        if remaining_energy < best_cost:
            break
    best_path_string = ' , '.join(map(str, best_path))
    return best_path_string, best_cost, np.sum(best_prize), remaining_energy

def graph_connectivity(Adj_matrix):
    num_nodes = len(Adj_matrix)
    visited = set()
    def dfs(node):
        visited.add(node)
        for neighbor in range(num_nodes):
            if Adj_matrix[node][neighbor] > 0 and neighbor not in visited:
                dfs(neighbor)
    dfs(0)
    return len(visited) == num_nodes


def main():
    x = int(input("Enter the width of the sensor network (in meters) [default: 2000]: "))
    y = int(input("Enter the length of the sensor network (in meters) [default: 2000]: "))
    N = int(input("Enter the number of sensor nodes (default: 100): "))
    Tr = int(input("Enter the transmission range in meters (default: 400): "))
    p = int(input("Enter the number of DNs (default: 50): "))
    q = int(input("Enter the maximum number of data packets each DN has (default: 1000): "))
    x = float(x)
    y = float(y)
    locations = [[random.uniform(0, float(x)), random.uniform(0, float(y))] for _ in range(N)]
    Adj_matrix = cdist(locations, locations)
    while not graph_connectivity(Adj_matrix):
        print("The network is not connected. Please try again.")
        sys.exit()
    budget = int(input("Enter the initial energy level of the robot (default 1000000): ") )
    prize = [random.random() * q for _ in range(N)]
    dn_indices = random.sample(range(N), p)
    dn_data_packets = [random.randint(1, q) for _ in range(p)]
    print("DNs and their data packets:")
    dn_info = [f"DN {dn_indices[i]}: {dn_data_packets[i]} data packets" for i in range(p)]
    print("\n".join(dn_info))
    algorithms = [
        ("Greedy 1", Greedy1_algorithm),
        ("Greedy 2", Greedy2_algorithm),
        ("MARL", MARL_algorithm)
    ]
    for algorithm_name, algorithm_fn in algorithms:
        print("\n{} Algorithm:".format(algorithm_name))
        path, cost, prize1, budget = algorithm_fn(Adj_matrix, prize,budget)
        print("Path:", path)
        print("Cost of the Path:", cost)
        print("Total Prizes Along the Path:", prize1)
        print("Remaning energy:",budget)
if __name__ == '__main__':
    main()
