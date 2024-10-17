from multiprocessing import Pool
import os
from multiprocessing.dummy import Pool as ThreadPool
from ortools.algorithms import pywrapknapsack_solver
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2
from scipy.spatial import distance_matrix
import numpy as np



def run_all_in_pool(func, input, use_multiprocessing=True):
    num_cpus = min(os.cpu_count(),30)
    pool_cls = (Pool if use_multiprocessing and num_cpus>1 else ThreadPool)
    with pool_cls(num_cpus) as pool:
        results = list(pool.imap(func, input))
    return results


def get_solution(data, manager, routing, solution):
    total_distance = 0
    total_route = []
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
        route_distance = 0.
        route_load = 0.
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            total_route.append(node_index)
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += data['distance_matrix'][manager.IndexToNode(previous_index)][manager.IndexToNode(index)]

        total_distance += route_distance
    return total_distance, total_route



def solve_cvrp_by_ortools(instance):
    CAPACITIES = {
        10: 20.,
        20: 30.,
        30: 40.,
        40: 40.,
        50: 40.,
        60: 50.,
        70: 50.,
        80: 50.,
        90: 50.,
        100: 50.
    }
    capacity = CAPACITIES[len(instance) - 1]
    depot, loc, demand = instance[0, :2].tolist(), instance[1:, :2].tolist(), (capacity * instance[1:, 2]).astype(np.int32).tolist()
    # Create the routing index manager.
    def cal_distance(loc):
        return distance_matrix(loc, loc)

    # Create and register a transit callback.
    def distance_callback(manager, data, from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    # Add Capacity constraint.
    def demand_callback(manager, data, from_index):
        """Returns the demand of the node."""
        # Convert from routing variable Index to demands NodeIndex.
        from_node = manager.IndexToNode(from_index)
        return data['demands'][from_node]

    data = {}
    data['distance_matrix'] = cal_distance(loc).tolist()
    data['demands'] = demand

    candidate_num_vechicles = list(range(3,21))

    for num_vechicles in candidate_num_vechicles:
        data['num_vehicles'] = num_vechicles
        data['vehicle_capacities'] = []
        for i in range(0, data['num_vehicles']):
            data['vehicle_capacities'].append(capacity)

        data['depot'] = 0

        manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                               data['num_vehicles'], data['depot'])
        routing = pywrapcp.RoutingModel(manager)
        call_back = lambda from_index, to_index: distance_callback(manager, data, from_index, to_index)
        transit_callback_index = routing.RegisterTransitCallback(call_back)
        # Define cost of each arc.
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        d_call_back = lambda from_index: demand_callback(manager, data, from_index)
        demand_callback_index = routing.RegisterUnaryTransitCallback(
            d_call_back)
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  # null capacity slack
            data['vehicle_capacities'],  # vehicle maximum capacities
            True,  # start cumul to zero
            'Capacity')

        # Setting first solution heuristic.
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
        search_parameters.time_limit.FromSeconds(1)

        # Solve the problem.
        solution = routing.SolveWithParameters(search_parameters)

        # Print solution on console.
        if solution:
            obj, route = get_solution(data, manager, routing, solution)
            return obj, route
    print("No solution found")
    return None, None


def solve_kp(instance, scale=1e4):
    instance *= scale
    values = instance[:,1].tolist()
    weights = [instance[:,0].tolist()]
    problem_size = instance.shape[0]
    if problem_size >= 20 and problem_size < 50:
        capacity = [6.25*scale]
    elif problem_size >= 50 and problem_size < 100:
        capacity = [12.5*scale]
    elif problem_size >= 100 and problem_size < 200:
        capacity = [25*scale]
    elif problem_size >= 200:
        capacity = [25*scale]
    else:
        raise NotImplementedError
    solver = pywrapknapsack_solver.KnapsackSolver(
        pywrapknapsack_solver.KnapsackSolver.
        KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER, 'Knapsack')

    solver.Init(values, weights, capacity)
    computed_value = solver.Solve() / scale
    packed_items = []
    packed_weights = []
    total_weight = 0
    for i in range(len(values)):
        if solver.BestSolutionContains(i):
            packed_items.append(i)
            packed_weights.append(weights[0][i]/scale)
            total_weight += weights[0][i]
    return computed_value, packed_items


if __name__=="__main__":
    instance = np.random.rand(21,2)
    solve_kp(instance)