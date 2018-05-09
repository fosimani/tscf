"""Trajectory recovery module"""
import itertools
import numpy as np

from scipy.optimize import linear_sum_assignment

from tracer.towers import TowersManager


class TrajectoryRecovery(object):
    """Upturn users trajectories from ash using aggregated data"""

    def __init__(
        self,
        number_users,
        towers,
        aggregated_data,
        vel_friction=0.9
    ):
        self.aggregated_data = aggregated_data
        self.towers = towers
        self.vel_friction = vel_friction

        self.number_users = number_users
        self.number_cycles = aggregated_data.shape[0]
        self.number_towers = aggregated_data.shape[1]

        self.towers_manager = TowersManager(towers, vel_friction=vel_friction)
        self.distances = self.towers_manager.generate_distances()

    def build_distribution_matrix(self):
        L = []

        for cycle_counts in self.aggregated_data:
            L.append(
                np.array(list(
                    itertools.chain(*(
                        [tower_index] * int(count)
                        for tower_index, count in enumerate(cycle_counts)
                    ))
                ))
            )

        self.L = np.array(L)

    def trajectory_recovery_generator(self):
        self.S = []
        self.C = [None]

        for cycle in range(self.number_cycles):
            if cycle == 0:
                self.S.append(np.random.permutation(self.L[0]))
            else:
                if cycle == 1:
                    #
                    # If it's on the night, we estimate the next location as the last one.
                    #
                    L_next_est = self.S[cycle - 1]
                else:
                    #
                    # During daylight, we estimate the next location taking into account
                    # the current users trajectory and direction.
                    #
                    L_next_est = []
                    for user in range(self.number_users):
                        direction = [
                            self.towers[self.S[cycle - 2][user]],
                            self.towers[self.S[cycle - 1][user]]
                        ]
                        new_point = \
                            self.towers_manager.get_new_point(direction)
                        l_next_est = \
                            self.towers_manager.get_nearest_tower(new_point)

                        L_next_est.append(l_next_est)

                    L_next_est = np.array(L_next_est)

                L_next = self.L[cycle]

                #
                # Calculate the cost matrix as the distance between the estimated tower and the
                # rest.
                #
                C_next = np.zeros((self.number_users, self.number_users))

                for i, l_next_est in enumerate(L_next_est):
                    for j, l_next in enumerate(L_next):
                        C_next[i, j] = self.distances[l_next, l_next_est]

                #
                # Append the cost matrix to the collection of cost matrices
                #
                self.C.append(C_next)

                #
                # Ref: https://docs.scipy.org/doc/scipy-0.18.1/reference/
                #  generated/scipy.optimize.linear_sum_assignment.html
                #
                # Solves the assignament problem of assiging the users the next
                # location taking into account the matrix of costs. The result
                # comes in the form of indexes, being the row_index the number of users
                # in this case, and the col_index the tower index. Therefore, the
                # col_index is the S_next we're looking for.
                #
                _, col_ind = linear_sum_assignment(C_next)

                self.S.append(L_next[col_ind])

        self.C = np.array(self.C)
        self.S = np.array(self.S)

        return {
            'recovered_costs': self.C,
            'recovered_trajectories': self.S,
        }

    def get_traces_common_elements(self, trace_1, trace_2):
        return np.sum(trace_1 == trace_2)

    def map_traces(self, real_traces):
        result = []
        used_traces = np.array([False for _ in real_traces])
        acc = []

        for trace in self.S.T:
            common_elements = np.array([
                self.get_traces_common_elements(trace, x) for x in real_traces
            ])
            common_elements[used_traces] = -1
            min_distance_index = np.argmax(common_elements)
            acc.append(common_elements[min_distance_index])
            used_traces[min_distance_index] = True
            result.append(min_distance_index)

        acc = np.array(acc)
        global_accuracy = np.sum(acc / self.number_cycles) / self.number_users
        return result, global_accuracy, acc
