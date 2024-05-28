import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances
import matplotlib.pyplot as plt

class PBC:
    """ ____________________________________________________________________________________________________
    Call the static method perform_clustering to perform clustering
        perform_clustering(distance_matrix, k=None, find_outliers=False, merge_option='largest_gap', n_clusters=None)

    Inputs:
        1) distance_matrix
            The distance matrix

        2) k: int, default=None
            Neighborhood size, k, is a positive integer.
            If (k<1 or k is None) the appropriate value for k is determined automatically.
            The default value is None.

        3) find_outliers: bool, default=False
            If True, the logical index of outliers is returned as the second output.
            If False, the second output is None.
            The default value is False.

        4) merge_option: {'no_merge', 'n_clusters', 'largest_gap'}, default='largest_gap'
            'no_merge' means that no merges is performed.
            'n_clusters' means that the number of merge phases is determined according to actual number of clusters
                specified by n_clusters argument.
            'largest_gap' means that the number of merge phased is determined according to largest cohesion gap method.
            The default value is 'largest_gap'.

        5) n_clusters: int, default=None
            The actual number of clusters. the value is ignored if merge_option != 'n_clusters'.
            The default value is None.

        Outputs: labels_pred, bi_outliers, o
            1) labels_pred
                The first output is the label vector,
            2) bi_outliers
                The second output is the logical index of outliers (None if find_outliers==False)
            3) out_dict
                The third output is a dict object which includes popularity_vector, backbone_threshold,
                outlier_threshold and etc.

        ____________________________________________________________________________________________________
        Call the static method test_pbc to apply the clustering method on an example dataset (mspsc5)
             test_pbc()
        Inputs: None
        Outputs: None (shows and saves plots for the dataset (mspsc5))
        ____________________________________________________________________________________________________ """

    default_rho = 0.8
    default_sigma = 0.001
    default_max_t = 100
    default_max_k = 100
    default_epsilon_for_t = 0.001
    default_threshold = 0.05

    @staticmethod
    def perform_clustering(distance_matrix, k=None, find_outliers=False, merge_option='largest_gap', n_clusters=None):
        rho, max_t, epsilon_for_t = PBC.default_rho, PBC.default_max_t, PBC.default_epsilon_for_t

        if k is None or k<1:
            k = PBC.estimate_appropriate_k_value(distance_matrix, epsilon_for_t=epsilon_for_t, max_t=-1)

        n_samples = distance_matrix.shape[0]
        k = min(max(k, 1), n_samples-1)

        unlabeled = -1
        np.fill_diagonal(distance_matrix, float('inf'))
        knn_indices = PBC.get_knn_indices(distance_matrix, k)

        reverse_knn_ranks = np.full((n_samples, k), k)
        for i in range(k):
            row_column_indices = np.argwhere(knn_indices[knn_indices[:,i], :] == np.arange(n_samples).reshape(-1,1))
            reverse_knn_ranks[row_column_indices[:, 0], i] = row_column_indices[:, 1]

        weight_vector = PBC.get_weight_vector(k+1)
        popularity_vector, t = PBC.get_popularity_vector(np.concatenate([knn_indices, np.arange(n_samples).reshape(-1, 1)], axis=1),
                                                         weight_vector, epsilon_for_t=epsilon_for_t, max_t=max_t)
        extended_weight_vector = np.concatenate([weight_vector[:-1], [0]])

        sort_order = np.argsort(popularity_vector)[::-1]
        reverse_sort_order = np.argsort(sort_order)
        sorted_popularity_vector = popularity_vector[sort_order]

        sorted_popularity_vector_cumsum = np.cumsum(sorted_popularity_vector)
        backbone_threshold, outlier_threshold = sorted_popularity_vector[np.count_nonzero(sorted_popularity_vector_cumsum <= rho * n_samples)], -1
        if find_outliers:
            sigma = PBC.default_sigma
            m = np.count_nonzero((n_samples-sorted_popularity_vector_cumsum) > sigma * n_samples) - 1
            largest_gap_index = np.argmax(sorted_popularity_vector[m:-1] - sorted_popularity_vector[m+1:])
            outlier_threshold = np.mean(sorted_popularity_vector[m+largest_gap_index:m+largest_gap_index + 1 + 1])

        adjacency_matrix = np.full((n_samples, n_samples), 0, dtype=int)
        bi_backbone_samples = popularity_vector >= backbone_threshold
        bi_nonbackbone_samples, n_backbone_samples = np.logical_not(bi_backbone_samples), np.count_nonzero(bi_backbone_samples)
        n_nonbackbone_samples = n_samples - n_backbone_samples
        knn_similarities_matrix = extended_weight_vector[np.maximum(np.arange(k), reverse_knn_ranks)]
        labels, current_label = np.full(n_samples, unlabeled), unlabeled
        for xi in sort_order[:n_backbone_samples]:
            if labels[xi] == unlabeled:
                current_label += 1
                labels[xi], queue = current_label, [xi]
                while len(queue) > 0:
                    xi = queue.pop(0)
                    bi_unlabeled_backbone_neighbors = np.logical_and(bi_backbone_samples[knn_indices[xi, :]], labels[knn_indices[xi, :]] == unlabeled)
                    bi_unlabeled_backbone_neighbors = np.logical_and(bi_unlabeled_backbone_neighbors, knn_similarities_matrix[xi, :] > 0)
                    neighbor_candidates_indices = knn_indices[xi, bi_unlabeled_backbone_neighbors]
                    neighbor_candidates_similarities = knn_similarities_matrix[xi, bi_unlabeled_backbone_neighbors].ravel()
                    n_neighbors = min(k, len(neighbor_candidates_indices))
                    i_neighbors = neighbor_candidates_indices[np.argpartition(-neighbor_candidates_similarities, n_neighbors - 1)[:n_neighbors]]
                    if len(i_neighbors) > 0:
                        labels[i_neighbors] = current_label
                        adjacency_matrix[xi, i_neighbors] = 1
                        queue.extend(list(i_neighbors))
        knn_indices_nonbackbone, reverse_knn_ranks_nonbackbone = knn_indices[sort_order[n_backbone_samples:], :], reverse_knn_ranks[sort_order[n_backbone_samples:], :]
        z = 0
        superior_score_matrix = extended_weight_vector[reverse_knn_ranks_nonbackbone] * popularity_vector[knn_indices_nonbackbone]
        superior_score_matrix[popularity_vector[knn_indices_nonbackbone] <= popularity_vector[sort_order[n_backbone_samples:]].reshape(-1, 1)] = z
        max_score, max_score_index = np.max(superior_score_matrix, axis=1), np.argmax(superior_score_matrix, axis=1)
        direct_superior = np.full(n_nonbackbone_samples, unlabeled)
        direct_superior[max_score != z] = np.take_along_axis(knn_indices_nonbackbone[max_score != z, :], max_score_index[max_score != z].reshape(-1,1), axis=1).ravel()
        i_faraway = sort_order[n_backbone_samples + np.argwhere(max_score == z).ravel()]
        if len(i_faraway)>0:
            distance_matrix_faraway = np.copy(distance_matrix[i_faraway, :])
            distance_matrix_faraway[reverse_sort_order[i_faraway].reshape(-1, 1) < reverse_sort_order] = float('inf')
            direct_superior[max_score == z] = np.argmin(distance_matrix_faraway, axis=1)
        for i in range(n_nonbackbone_samples):
            labels[sort_order[n_backbone_samples+i]] = labels[direct_superior[i]]
            adjacency_matrix[sort_order[n_backbone_samples+i], direct_superior[i]] = 2
        assert np.count_nonzero(labels == unlabeled) == 0, 'some samples are not assigned to clusters.'
        predicted_labels = labels.copy()
        labels_pred, cluster_indices_matrix, cohesion_vector, count_vector = \
            PBC.perform_merging_phase(predicted_labels, popularity_vector, extended_weight_vector, knn_indices, reverse_knn_ranks, merge_option, n_clusters)
        out_dict = dict(popularity_vector=popularity_vector, reverse_knn_ranks=reverse_knn_ranks, backbone_threshold=backbone_threshold,
                        outlier_threshold=outlier_threshold, knn_indices=knn_indices, k=k, t=t, labels=labels_pred, adjacency_matrix=adjacency_matrix,
                        cohesion_vector=cohesion_vector, count_vector=count_vector, cluster_indices_matrix=cluster_indices_matrix)
        bi_outliers = popularity_vector <= outlier_threshold if find_outliers else None
        return labels_pred, bi_outliers, out_dict

    @staticmethod
    def test_pbc():
        data = pd.read_csv('mspsc5.csv', sep=',', header=None).to_numpy()
        samples, labels = data[:,:-1], data[:,-1]
        PBC.show_plot(samples, labels, 'original data')
        distance_matrix = pairwise_distances(samples)

        labels_pred, *_ = PBC.perform_clustering(distance_matrix, k=50)
        PBC.show_plot(samples, labels_pred, "k=50, merge_option not given (default is 'largest_gap')")

        labels_pred, bi_outliers, _ = PBC.perform_clustering(distance_matrix, k=50, find_outliers=True)
        PBC.show_plot(samples, labels_pred, "k=50, find_outliers=True (red stars are outliers)",
                      bi_outliers=bi_outliers)

        labels_pred, *_ = PBC.perform_clustering(distance_matrix, k=50, merge_option='n_clusters', n_clusters=13)
        PBC.show_plot(samples, labels_pred, "k=50, merge_option='n_clusters'")

        labels_pred, *_ = PBC.perform_clustering(distance_matrix, k=100, merge_option='no_merge')
        PBC.show_plot(samples, labels_pred, "k=100, merge_option='no_merge'")

        labels_pred, *_ = PBC.perform_clustering(distance_matrix, merge_option='n_clusters', n_clusters=13)
        PBC.show_plot(samples, labels_pred, "without k parameter, merge_option='n_clusters'")

    @staticmethod
    def show_plot(samples, labels, caption, bi_outliers=None):
        fig = plt.figure(figsize=(8, 6))
        if bi_outliers is not None:
            plt.scatter(samples[bi_outliers, 0], samples[bi_outliers, 1], s=10, marker='*', c='r')
            bi_inliers = np.logical_not(bi_outliers)
            plt.scatter(samples[bi_inliers, 0], samples[bi_inliers, 1], s=5, c=labels[bi_inliers])
        else:
            plt.scatter(samples[:, 0], samples[:, 1], s=5, c=labels)
        plt.title(caption)
        plt.tick_params(axis='x', which='both', left=False, bottom=False, top=False, labelbottom=False, right=False)
        plt.xticks([], [])
        plt.yticks([], [])
        plt.show()
        fig.savefig(caption+'.png', format='png', dpi=300, bbox_inches='tight')


    @staticmethod
    def get_knn_indices(distance_matrix, k):
        knn_indices_unsorted = np.argpartition(distance_matrix, k - 1, axis=1)[:, :k]
        knn_distances_unsorted = np.take_along_axis(distance_matrix, knn_indices_unsorted, axis=1)
        return np.take_along_axis(knn_indices_unsorted, np.argsort(knn_distances_unsorted, axis=1), axis=1)

    @staticmethod
    def get_weight_vector(k):
        scale = 20
        minimum_value = 10 ** (-1 - (k / scale))
        l, u, convergence_threshold = 0, 1, 1e-6
        while u - l > convergence_threshold:
            m = (l + u) / 2
            error = minimum_value * (1 - m ** k) - m ** (k - 1) * (1 - m)
            if error > 0:
                l = m
            elif error < 0:
                u = m
            else:
                break
        ratio = (l + u) / 2
        a = (1 - ratio) / (1 - ratio ** k)
        weight_vector = a * np.power(ratio, range(k))
        return weight_vector / sum(weight_vector)

    @staticmethod
    def get_popularity_vector(knn_indices, weight_vector, epsilon_for_t=None, max_t=None):
        if epsilon_for_t is None:
            epsilon_for_t = PBC.default_epsilon_for_t
        if max_t is None:
            max_t = PBC.default_max_t

        assert knn_indices.shape[1] == weight_vector.size, "Size error!"
        assert epsilon_for_t > 0 or (0 < max_t < float('inf'))
        n_samples = knn_indices.shape[0]
        old_popularity_vector = np.ones(n_samples)
        done, t = False, 0
        while not done:
            weight_matrix = weight_vector * old_popularity_vector.reshape(-1, 1)
            popularity_vector = np.zeros(n_samples)
            for i in range(n_samples):
                popularity_vector[knn_indices[i, :]] += weight_matrix[i, :]
            difference = np.mean(np.abs(popularity_vector - old_popularity_vector))
            old_popularity_vector = popularity_vector
            t += 1
            done = (t == max_t and max_t >= 0) or (difference <= epsilon_for_t and epsilon_for_t > 0)
        return popularity_vector, t

    @staticmethod
    def perform_merging_phase(labels, popularity_vector, extended_weight_vector, knn_indices, reverse_knn_ranks, merge_option, n_clusters_true):
        cohesion_matrix, count_matrix = PBC.get_cohesion_matrix(labels, popularity_vector, extended_weight_vector, knn_indices, reverse_knn_ranks)
        cluster_indices_matrix, cohesion_vector, count_vector = PBC.perform_successive_merges_until_cohesion_is_zero(labels, cohesion_matrix, count_matrix)
        n_merge_phases = PBC.estimate_the_number_of_merges_required(merge_option, cohesion_vector, n_clusters_true)
        n_merge_phases = max(0, min(n_merge_phases, np.shape(cluster_indices_matrix)[1] - 1))
        labels_pred = np.copy(cluster_indices_matrix[:, n_merge_phases])
        return labels_pred, cluster_indices_matrix, cohesion_vector, count_vector

    @staticmethod
    def get_cohesion_matrix(labels, popularity_vector, extended_weight_vector, knn_indices, reverse_knn_ranks):
        knn_labels = labels[knn_indices]
        knn_tendencies = extended_weight_vector[:-1] * popularity_vector[knn_indices]
        reverse_knn_tendencies = extended_weight_vector[reverse_knn_ranks] * popularity_vector.reshape(-1,1)
        knn_mutual_tendencies = np.minimum(knn_tendencies, reverse_knn_tendencies)
        label_values = np.unique(labels)
        assert np.all(label_values == [*range(len(label_values))]), 'size error!'
        n_clusters = np.max(labels) + 1
        cohesion_matrix, count_matrix = np.zeros((n_clusters, n_clusters)), np.zeros((n_clusters, n_clusters))
        for ci in range(n_clusters):
            knn_mutual_tendencies_ci = knn_mutual_tendencies[labels == ci, :]
            knn_labels_ci = knn_labels[labels == ci, :]
            for cj in range(ci + 1, n_clusters):
                mutual_tendencies_ci_cj = knn_mutual_tendencies_ci[knn_labels_ci == cj].ravel()
                mutual_tendencies_ci_cj = mutual_tendencies_ci_cj[mutual_tendencies_ci_cj>0]
                count_matrix[ci, cj] = count_matrix[cj,ci] = len(mutual_tendencies_ci_cj)
                cohesion_matrix[ci,cj] = cohesion_matrix[cj,ci] = np.mean(mutual_tendencies_ci_cj) if count_matrix[ci, cj] > 0 else 0
        return cohesion_matrix, count_matrix

    @staticmethod
    def perform_successive_merges_until_cohesion_is_zero(labels, cohesion_matrix, count_matrix):
        new_labels = np.copy(labels)
        n_clusters, index, done = cohesion_matrix.shape[0], 0, False
        cluster_indices_matrix = np.zeros((new_labels.size, n_clusters))
        cohesion_vector, count_vector = np.zeros(n_clusters), np.zeros(n_clusters)
        while not done:
            cluster_indices_matrix[:, index] = np.copy(new_labels)
            cohesion_vector[index] = np.max(cohesion_matrix[np.triu_indices(n_clusters, 1)]) if n_clusters>1 else 0
            done = n_clusters == 1 or cohesion_vector[index] == 0
            if not done:
                rows_indices, columns_indices = np.where(cohesion_matrix == cohesion_vector[index])
                ci, cj = rows_indices[0], columns_indices[0]
                count_vector[index] = count_matrix[ci, cj]
                new_labels, cohesion_matrix, count_matrix = PBC.merge_two_clusters(new_labels, cohesion_matrix, count_matrix, ci, cj)
                n_clusters, index = n_clusters-1, index+1
        return cluster_indices_matrix, cohesion_vector, count_vector

    @staticmethod
    def merge_two_clusters(labels, cohesion_matrix, count_matrix, ci, cj):
        new_labels = np.copy(labels)
        assert ci != cj, 'ci must be different from cj'
        ci, cj = min(ci, cj), max(ci, cj)
        new_labels[labels == cj] = ci
        new_labels[labels > cj] -= 1

        prod_matrix = cohesion_matrix * count_matrix
        prod_matrix[ci, :] += prod_matrix[cj, :]
        count_matrix[ci,:] += count_matrix[cj,:]
        prod_matrix, count_matrix = np.delete(prod_matrix, cj, axis=0), np.delete(count_matrix, cj, axis=0)

        prod_matrix[:, ci] += prod_matrix[:, cj]
        count_matrix[:, ci] += count_matrix[:, cj]
        prod_matrix, count_matrix = np.delete(prod_matrix, cj, axis=1), np.delete(count_matrix, cj, axis=1)
        count_matrix[ci, ci], prod_matrix[ci,ci] = 0, 0
        mask = count_matrix == 0
        count_matrix[mask] = 1
        cohesion_matrix = np.divide(prod_matrix, count_matrix)
        count_matrix[mask] = 0

        return new_labels, cohesion_matrix, count_matrix

    @staticmethod
    def estimate_the_number_of_merges_required(merge_option, cohesion_vector, n_clusters_true):
        epsilon = 1e-6
        n_merges_required = 0
        assert (merge_option == 'no_merge' or merge_option == 'n_clusters' or merge_option == 'largest_gap'), \
            "invalid merge_option: merge_option must be one of these three values: 'no_merge', 'n_clusters', 'largest_gap'."
        n_clusters_pred = np.size(cohesion_vector)
        if merge_option == 'n_clusters':
            n_merges_required = max(0, n_clusters_pred - n_clusters_true)
            indices = np.argwhere(cohesion_vector == 0).ravel()
            if len(indices) > 0:
                n_merges_required = min(indices[0], n_merges_required)
        if merge_option == 'largest_gap':
            indices = np.argwhere(cohesion_vector <= epsilon).ravel()
            c_vector = cohesion_vector[: indices[0] if len(indices)>0 else len(cohesion_vector)-1]
            n_merges_required = len(c_vector)
            if len(c_vector) > 1:
                n_merges_required = np.argmax(c_vector[:-1] - c_vector[1:]) + 1
        return n_merges_required


    def get_rank_matrix(distance_matrix, k):
        # R(i, j) is the rank of j in the list of nearest neighbors of i, considering r=0 for the zeroth nearest neighbor (the point itself)
        distance_matrix_copy = np.copy(distance_matrix)
        n_samples = distance_matrix_copy.shape[0]
        np.fill_diagonal(distance_matrix_copy, float('inf'))
        knn_indices_unsorted = np.argpartition(distance_matrix_copy, k - 1, axis=1)[:, :k]
        knn_distances_unsorted = np.take_along_axis(distance_matrix_copy, knn_indices_unsorted, axis=1)
        knn_indices = np.take_along_axis(knn_indices_unsorted, np.argsort(knn_distances_unsorted, axis=1), axis=1)
        rank_matrix = np.full((n_samples,n_samples),k+1,dtype=float)
        np.put_along_axis(rank_matrix, np.concatenate([np.arange(n_samples).reshape(-1, 1), knn_indices], axis=1), np.arange(0, k + 1).reshape(1,-1), axis=1)
        return rank_matrix

    @staticmethod
    def get_indices_of_backbone_halo_and_outlier_samples(popularity_vector, backbone_threshold, outlier_threshold):
        bi_backbone_samples = popularity_vector >= backbone_threshold
        bi_nonbackbone_samples = np.logical_not(bi_backbone_samples)
        bi_outlier_samples = popularity_vector <= outlier_threshold
        bi_halo_samples = np.logical_not(np.logical_or(bi_backbone_samples, bi_outlier_samples))
        return bi_backbone_samples, bi_halo_samples, bi_outlier_samples, bi_nonbackbone_samples

    @staticmethod
    def estimate_appropriate_k_value(distance_matrix, epsilon_for_t=None, max_t=None, threshold=None):
        if threshold is None:
            threshold = PBC.default_threshold
        if max_t is None:
            max_t = PBC.default_max_t
        if epsilon_for_t is None:
            epsilon_for_t = PBC.default_epsilon_for_t

        n_samples = distance_matrix.shape[0]
        max_k = min(PBC.default_max_k, n_samples - 1)
        np.fill_diagonal(distance_matrix, float('inf'))
        knn_indices_for_max_k = PBC.get_knn_indices(distance_matrix, max_k)

        bi_backbone_samples_old = np.full(n_samples, False)
        for k in range(1, max_k+1):
            weight_vector = PBC.get_weight_vector(k + 1)
            popularity_vector, _ = PBC.get_popularity_vector(np.concatenate([knn_indices_for_max_k[:, :k], np.arange(n_samples).reshape(-1, 1)], axis=1),
                weight_vector, max_t=max_t, epsilon_for_t=epsilon_for_t)
            sorted_popularity_vector = np.sort(popularity_vector)[::-1]
            bi_backbone_samples_new = popularity_vector >= sorted_popularity_vector[np.count_nonzero(np.cumsum(sorted_popularity_vector) <= PBC.default_rho * n_samples)]
            change_fraction = np.count_nonzero(bi_backbone_samples_old != bi_backbone_samples_new)/n_samples
            bi_backbone_samples_old = bi_backbone_samples_new
            if change_fraction <= threshold:
                return k
        return max_k
   

    def get_backbone_and_outlier_threshold(sorted_popularity_vector:np.ndarray, backbone_method=None, rho=None, mark_outliers=False):
        n_samples = len(sorted_popularity_vector)

        sorted_popularity_vector_cumsum = np.cumsum(sorted_popularity_vector)
        backbone_threshold, outlier_threshold = sorted_popularity_vector[np.count_nonzero(sorted_popularity_vector_cumsum <= rho * n_samples)], -1
        n_backbone_samples = np.count_nonzero(sorted_popularity_vector > backbone_threshold)
        if mark_outliers:
            rho2 = PBC.default_sigma
            gamma = np.count_nonzero(sorted_popularity_vector_cumsum <= (1-rho2) * n_samples)-1
            largest_gap_index = np.argmax(sorted_popularity_vector[gamma:-1] - sorted_popularity_vector[gamma+1:])
            outlier_threshold = np.mean(sorted_popularity_vector[gamma+largest_gap_index:gamma+largest_gap_index + 1 + 1])
        return backbone_threshold, outlier_threshold

if __name__ == '__main__':
    PBC.test_pbc()