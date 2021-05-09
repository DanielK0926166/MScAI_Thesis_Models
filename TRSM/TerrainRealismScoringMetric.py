# -*- coding: utf-8 -*-
"""
@author: daniel.kiss

This file contains the main system responsible for evaluating the realism accuracy score of generated heightmaps 
by comparing them to real world heightmaps.

Note:
Code that has been commented out were previously used during the experimentation phase but are no longer needed for the final model
"""

import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import data_config as config


import matplotlib.pyplot as plt
import numpy as np
import time
import glob
import enum
import pickle
import shutil

from scipy.stats import wasserstein_distance
import scipy.cluster.hierarchy as sch

import concurrent.futures
import skimage.io

import json
import math

import sys
sys.setrecursionlimit(10000)

# Linkage type, different ways of calculating the distance between clusters
class LINKAGE(enum.Enum):
    COMPLETE = 0
    SINGLE   = 1
    AVERAGE  = 2
    WARD     = 3


# This function is executed on multiple processors in paralell
def initial_proxy_matrix_distance_calculator_thread(index, start_index, end_index, num_data_instances, train_data):
    group_size = end_index - start_index + 1
    results = np.zeros((group_size, num_data_instances))
    print("Process: {} started on {}".format(index, os.getpid()))
    for x in range(group_size):
        data_x = start_index + x
        hist_one = train_data[data_x]
        for y in range(data_x+1, num_data_instances):
            hist_two = train_data[y]
            w_dist = wasserstein_distance(hist_one, hist_two)
            results[x][y] = w_dist
    print("Process: {} ended".format(index))
    return (index, results)

# This function is executed on multiple processors in paralell
def within_cluster_variance_calculator_thread(index, start_index, end_index, cluster, proximity_matrix, within_dist_with):
    largest_diff = 0.0
    print("Process {} has started on {}".format(index, os.getpid()))
    for c in range(start_index, end_index):
        cluster_dupe = cluster[:]
        del cluster_dupe[c]
    
        within_dist_without = 0.0
        total_dist = []
        for i in range(0, len(cluster_dupe)-1):
            for k in range(i+1, len(cluster_dupe)):
                total_dist.append(proximity_matrix[cluster_dupe[i]][cluster_dupe[k]])  
        within_dist_without = np.nanmean(total_dist)

        diff = within_dist_with - within_dist_without
        if diff > largest_diff:
            largest_diff = diff
    print("Process {} has ended. largest_diff: {}".format(index, largest_diff))
    return largest_diff



# The main TRSM class
class My_Terrain_Realism_Scoring_Metric_Class:
    def __init__(self, num_bins, linkage_type):
        self.HISTOGRAM_FILENAME              = "cached_histogram_data.npy"
        self.TRAINING_IMG_NAMES_FILENAME     = "cached_img_file_names.npy"
        self.INITIAL_PROXY_MATRIX_FILE_NAME  = "cached_proximity_matrix.npy"
        self.MERGE_HISTORY_FILE_NAME         = "cached_merge_history.npy"
        self.MERGE_CLUSTER_HISTORY_FILE_NAME = "cached_merge_cluster_history.npy"
        self.CLUSTER_GROUPS_FILE_NAME        = "cached_cluster_groups.npy"
        self.CLUSTER_VARIANCE_FILE_NAME      = "cached_cluster_variance.npy"
        
        self.CLUSTER_INFO_FOLDER_NAME = "Cluster_Info"
        
        self.training_data      = None
        self.training_filenames = None
        self.proximity_matrix   = None
        self.merge_history      = None
        self.cluster_history    = None
        self.clusters           = None
        
        self.NUM_BINS     = num_bins
        self.LINKAGE_TYPE = linkage_type
        pass

    ##########################################################
    ##########################################################
    # PUBLIC FUNCTIONS
    ##########################################################
    
    def generate_histogram_data(self, folder_path, force_rebuild_histogram):
        """
        This function either generates or loads the already generated histogram data
        If the training data has changed, force_rebuild_histogram should be set to True, otherwise to False
        
        The process loads all training data images and extracts the histogram information
        """
        self.SUB_FOLDER_PATH = folder_path
        # create histograms
        if not os.path.isfile(self.HISTOGRAM_FILENAME) or force_rebuild_histogram:
            print("Rebuilding Histogram Data")
            # find image data
            glob_path = "{}\{}\{}\*.png".format(config.BASE_DATA_PATH, config.RESULT_FOLDER, folder_path)
            image_paths_list = glob.glob(glob_path)
            self.training_data, self.training_filenames = self._create_and_save_training_data_histograms(self.NUM_BINS, image_paths_list)
        else:
            print("Loading Cached Historgram Data")
            self.training_data, self.training_filenames = self._load_all_histograms()
        print("Process Complete")
        pass

    def generate_proximity_matrix(self, force_rebuild_init_proxy):
        """
        This function either generates or loads the already generated proximity matrix
        If the training data has changed, force_rebuild_init_proxy should be set to True, otherwise to False
        
        The process calculates each training data instance's Wasserstein distance to all other training data instances.
        02 complexity. Multiprocessor usage is utalised in order to speed up calculations for large datasets
        """        
        self.num_data_instances = self.training_data.shape[0]
        print("Num Data Instances: {}".format(self.num_data_instances))
        self.proximity_matrix = np.empty((self.num_data_instances, self.num_data_instances))
        self.proximity_matrix[:] = np.NaN
        if not os.path.isfile(self.INITIAL_PROXY_MATRIX_FILE_NAME) or force_rebuild_init_proxy:
            print("Rebuilding Initial Proximity Matrix")
            self._re_calculate_proximity_matrix()
        else:
            print("Loading Initial Proximity Matrix")
            self._load_initial_proximity_matrix()
            
        print("Process Complete")
        pass
    
    def generate_merge_history_hierarchy(self, force_rebuild_linkage):
        """
        This function either generates or loads the already generated cluster merge and linkage history
        If the training data has changed, force_rebuild_linkage should be set to True, otherwise to False
        """
        if not os.path.isfile(self.MERGE_HISTORY_FILE_NAME) or not os.path.isfile(self.MERGE_CLUSTER_HISTORY_FILE_NAME) or force_rebuild_linkage:
            # copy the matrix and add a cluster index to the front
            self._merge_clusters(self.LINKAGE_TYPE)
        else:
            self._load_merge_history()
        print("Process Complete")
        pass
    
    def display_dendrogram(self, threshold):        
        plt.figure()
        #plt.title('Hierarchical Clustering Dendrogram')
        plt.ylabel('Distance')
        plt.xlabel('Clusters')
        sch.dendrogram(
            self.merge_history,
            orientation='top',
            no_labels=True,
            color_threshold=threshold,
            above_threshold_color="b"
        )
        
        plt.axhline(y=threshold, color='r', linestyle='--')
        plt.show()
        pass
    
    def display_histogram(self, data):
        n, bins, patches = plt.hist(x=data, bins='auto', alpha=0.7, rwidth=0.85)
        plt.grid(axis='y', alpha=0.75)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title('Within-Cluster Distribution')
        plt.text(23, 45, r'$\mu=15, b=3$')
        maxfreq = n.max()
        # Set a clean upper y-axis limit.
        plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
        plt.show()
        pass
    
    def generate_clusters_for_prediction(self, num_clusters, force_rebuild_cluster_groups):
        """
        This function either generates or loads the already generated cluster groups
        If the training data or num_clusters has changed, force_rebuild_cluster_groups should be set to True, otherwise to False
        """
        self.NUM_CLUSTERS = num_clusters
        
        if not os.path.isfile(self.CLUSTER_GROUPS_FILE_NAME) or force_rebuild_cluster_groups:
            self._create_clusters_from_training_data(num_clusters)
        else:
            self._load_cluster_groups()
        
        
        for c in range(len(self.clusters)):
            cluster = self.clusters[c]
            print("Cluster {} size: {}".format(c, len(cluster)))
        
        
#        if not os.path.isfile(self.CLUSTER_VARIANCE_FILE_NAME) or force_rebuild_cluster_groups:
#            self.calculate_within_cluster_variance()
#        else:
#            self._load_within_cluster_variance()
        
#        for c in range(len(self.clusters)):
#            print("Cluster: {}'s Within-Cluster Variance: {}".format(c, self.within_cluster_variance[c]))
#        
        self.calculate_within_cluster_distance()
        pass
    
    def predict(self, file_path):
        """
        Used to predict the group and accuracy of an image
        """
        return self._predict_cluster_and_accuracy(file_path, self.NUM_BINS, self.NUM_CLUSTERS)
    
    ##########################################################
    ##########################################################
    # PRIVATE FUNCTIONS
    ##########################################################
    
    ##########################################################
    # HISTOGRAM
    def _create_and_save_training_data_histograms(self, num_bins, image_paths_list):
        all_imagenames = []
        all_historgrams = []
        for image_path in image_paths_list:
            im_name = image_path.split('\\')[-1]
            config.PrintDebug("Processing: {}".format(im_name))
            
            # load each image file and change the pixel range to be from 0 - 1.
            image = skimage.io.imread(fname=image_path, as_gray=True) / config.U16_MAX_VAL
            # convert image to histogram
            histogram, bin_edges = np.histogram(image, bins=num_bins, range=(0, 1))
            # cache histogram into array
            all_historgrams.append(histogram)
            all_imagenames.append(im_name)
        
        # convert to numpy array and save it to file
        all_historgrams = np.array(all_historgrams)
        with open(self.HISTOGRAM_FILENAME, 'wb') as f:
            config.PrintDebug("Saving Histogram Data Into {}".format(self.HISTOGRAM_FILENAME))
            np.save(f, all_historgrams)
        config.PrintDebug("Process complete. Saved: {} Rows Of Data".format(all_historgrams.shape[0]))
        
        all_imagenames = np.array(all_imagenames)
        with open(self.TRAINING_IMG_NAMES_FILENAME, "wb") as f:
            config.PrintDebug("Saving Training Image Filenames Into {}".format(self.TRAINING_IMG_NAMES_FILENAME))
            np.save(f, all_imagenames)
        
        return (all_historgrams, all_imagenames)

    def _load_all_histograms(self):
        # load cached histogram data into numpy array
        np_array = None
        with open(self.HISTOGRAM_FILENAME, 'rb') as f:
            np_array = np.load(f)
        np_filenames = None
        with open(self.TRAINING_IMG_NAMES_FILENAME, 'rb') as f:
            np_filenames = np.load(f)
        return (np_array, np_filenames)
    ##########################################################

    ##########################################################
    # PRIOXIMITY MATRIX
    def _re_calculate_proximity_matrix(self):
        start_time = time.time()
        
        num_processes = config.MAX_AVAILABLE_CPU_CORES * 3 # should be larger then available processors because the work is not evenly spread
        
        # Spread the work out around all the processes
        # Each process will have around the same number of data instances to handle, but higher index value has less calculations to do then lower index value data
        range_indices = np.array(range(0, self.num_data_instances))
        divided_indices = np.array_split(range_indices, num_processes)
        
        # Execute multi-processor functions
        with concurrent.futures.ProcessPoolExecutor(max_workers=config.MAX_AVAILABLE_CPU_CORES) as executor:
            results = [executor.submit(initial_proxy_matrix_distance_calculator_thread, index, divided_indices[index][0], divided_indices[index][-1], self.num_data_instances, self.training_data) for index in range(num_processes)]   
            
            # collect the results as the processes complete and fill out the proximity matrix
            for f in concurrent.futures.as_completed(results):
                index, results = f.result()
                for x in range(results.shape[0]):
                    x_pos = divided_indices[index][0] + x
                    for y in range(x_pos+1, self.num_data_instances):
                        self.proximity_matrix[x_pos][y] = results[x][y]
                        self.proximity_matrix[y][x_pos] = results[x][y]
        
        config.PrintDebug("Completed in {:.2f} seconds. Sum {}".format((time.time() - start_time), np.nansum(self.proximity_matrix)))
        
        with open(self.INITIAL_PROXY_MATRIX_FILE_NAME, 'wb') as f:
            config.PrintDebug("Saving Initial Proximity Matrix Into {}".format(self.INITIAL_PROXY_MATRIX_FILE_NAME))
            np.save(f, self.proximity_matrix)
        pass
    
    def _load_initial_proximity_matrix(self):
        with open(self.INITIAL_PROXY_MATRIX_FILE_NAME, 'rb') as f:
            config.PrintDebug("Loading Initial Proximity Matrix From {}".format(self.INITIAL_PROXY_MATRIX_FILE_NAME))
            self.proximity_matrix = np.load(f)
        pass
    
    ##########################################################
    # HIERARCHICAL CLUSTER MERGING
    def _find_smallest_index(self, proximity_matrix):
        # Finds the smallest distance in the proximity matrix. Ignores NaN-s
        # The index returned is for position in a 1D version of the 2D matrix
        # so the 1D position has to be converted to row x column coordinates
        width = proximity_matrix.shape[0]
        index = np.nanargmin(proximity_matrix)
        row = int(index / width)
        col = index % width
        return [row, col]


    def _calculate_subset_of_proximity_matrix(self, linkage_type, all_current_indices, combined_cluster_indices, cluster_counters):
        # calculates the distances of the newly merged cluster to all other clusters
        distances = []
        for index in all_current_indices:
            dist = self._get_cluster_distance_for_linkage(linkage_type, self.proximity_matrix, combined_cluster_indices, cluster_counters[index])
            distances.append(dist)
        return np.array(distances)

    def _get_cluster_distance_for_linkage(self, linkage_type, proxy_matrix, cluster_one, cluster_two):
        # based on the linkage type, it calculates the distance between clusters
        if linkage_type == LINKAGE.COMPLETE:
            max_dist = 0.0
            for c_one in cluster_one:
                for c_two in cluster_two:
                    d = proxy_matrix[c_one][c_two]
                    if d > max_dist:
                        max_dist = d
            return max_dist
        
        if linkage_type == LINKAGE.SINGLE:
            min_dist = float("inf")
            for c_one in cluster_one:
                for c_two in cluster_two:
                    d = proxy_matrix[c_one][c_two]
                    if d < min_dist:
                        min_dist = d
            return min_dist
        
        if linkage_type == LINKAGE.AVERAGE:
            clusters_sizes = len(cluster_one) + len(cluster_two)
            sum_dist = 0.0
            for c_one in cluster_one:
                for c_two in cluster_two:
                    sum_dist += proxy_matrix[c_one][c_two]
            return (sum_dist / clusters_sizes)
        
        if linkage_type == LINKAGE.WARD:
            cluster_one_list = list(cluster_one)
            cluster_two_list = list(cluster_two)
            
            cluster_one_size = len(cluster_one_list)
            cluster_two_size = len(cluster_two_list)
            
            # in cluster average distance of cluster one
            clust_one_average = 0.0
            for i in range(0, cluster_one_size-1):
                for k in range(i, cluster_one_size):
                    clust_one_average += proxy_matrix[cluster_one_list[i]][cluster_one_list[k]]
            clust_one_average /= cluster_one_size
            # in cluster average distance of cluster two
            clust_two_average = 0.0
            for i in range(0, cluster_two_size-1):
                for k in range(i, cluster_two_size):
                    clust_two_average += proxy_matrix[cluster_two_list[i]][cluster_two_list[k]]
            clust_two_average /= cluster_two_size
            
            
            # in cluster average distance of merged cluster
            merged_clusters = cluster_one_list + cluster_two_list
            merged_cluster_size = len(merged_clusters)
            sum_dist = 0.0
            for x in range(0, merged_cluster_size-1):
                for y in range(x+1, merged_cluster_size):
                    sum_dist += proxy_matrix[merged_clusters[x]][merged_clusters[y]]
            sum_dist /= merged_cluster_size
            
            if not math.isnan(clust_one_average):
                sum_dist -= clust_one_average
            if not math.isnan(clust_two_average):
                sum_dist -= clust_two_average
            return sum_dist
        
        pass

    def _merge_clusters(self, linkage_type):        
        self.merge_history = []
        
        proxy_matrix_dupe = np.array(self.proximity_matrix, copy=True) # create duplicate of the initial proximity matrix
        current_largest_cluster_index = proxy_matrix_dupe.shape[0]     # rolling cluster index, each new cluster gets a new index
        row_indices = np.arange(current_largest_cluster_index)         # helper array to help refer back to the original histogram
        
        # Create an array of arrays tracking which elements are in each cluster.
        # At the start, each cluster index only contains itself
        self.cluster_history = []
        for i in range(current_largest_cluster_index):
            self.cluster_history.append([i])
        
        while proxy_matrix_dupe.shape[0] > 1:
            start_time = time.time()
            print("Meging Next Closest Clusters. {} Clusters are remaining".format(proxy_matrix_dupe.shape[0]))
            
            # 1. Find the next closes cluster
            smallest_index = self._find_smallest_index(proxy_matrix_dupe) # returns the row and column indices of the closest cluster
            smallest_distance = proxy_matrix_dupe[smallest_index[0]][smallest_index[1]] # gets the actual distance to this closest cluster
            smallest_original_index = [row_indices[smallest_index[0]], row_indices[smallest_index[1]]] # gets the original indices of this cluster referencing the original training data position
            
            # 2. Get which elements are in both clusters being merged
            cluster_one = self.cluster_history[smallest_original_index[0]]
            cluster_two = self.cluster_history[smallest_original_index[1]]
            both_clusters = cluster_one + cluster_two
            self.cluster_history.append(both_clusters) # keep track of all the elements in this new cluster
            
            # 3. delete the data belonging to the clusters being merged.
            proxy_matrix_dupe = np.delete(proxy_matrix_dupe, [smallest_index[0], smallest_index[1]], axis=0) # delete rows
            proxy_matrix_dupe = np.delete(proxy_matrix_dupe, [smallest_index[0], smallest_index[1]], axis=1) # delete cols
            # delete the row indices as well to make sure that the found index will continue referencing the original index
            row_indices = np.delete(row_indices, [smallest_index[0], smallest_index[1]])
            
            # 4. calculate the distances of this new cluster against all other clusters.
            # This has to happen after the deletion so we don't waste time
            distances = self._calculate_subset_of_proximity_matrix(linkage_type, row_indices, both_clusters, self.cluster_history)
            
            # 5. Add the distances to the proximity matrix, both as column and as row
            proxy_matrix_dupe = np.vstack((proxy_matrix_dupe, distances)) # add new row to 2d matrix
            distances = np.append(distances, np.NaN) # add an extra NaN to the end of the list, this will be for the distance to itself
            proxy_matrix_dupe = np.column_stack((proxy_matrix_dupe, distances)) # add new column to 2d matrix
            
            
            # 6. Keep tracking the index of this new cluster
            row_indices = np.append(row_indices, current_largest_cluster_index)
            current_largest_cluster_index += 1
            
            # 7. add result to merge_history
            # The merge history for each merge is an array with size 4:
            # Index 0 and 1 in this array references the indices of the clusters that got merged
            # Index 2 is the distance between the merged clusters
            # Index 3 is the size of the new cluster, how many training data instances it contains
            cluster_size = len(both_clusters)
            res = [smallest_original_index[0], smallest_original_index[1], smallest_distance, cluster_size]
            self.merge_history.append(res)
            
            config.PrintDebug("Merging {} with {} completed in {} seconds. Distance: {}, Cluster Size: {}".format(smallest_original_index[0], smallest_original_index[1], (time.time() - start_time), smallest_distance, cluster_size))
             
        self.merge_history = np.array(self.merge_history)
        with open(self.MERGE_HISTORY_FILE_NAME, 'wb') as f:
            config.PrintDebug("Saving Merge History Into {}".format(self.MERGE_HISTORY_FILE_NAME))
            np.save(f, self.merge_history)
        
        with open(self.MERGE_CLUSTER_HISTORY_FILE_NAME, 'wb') as f:
            config.PrintDebug("Saving Merge Cluster History Into {}".format(self.MERGE_CLUSTER_HISTORY_FILE_NAME))
            pickle.dump(self.cluster_history, f)
        pass
    
    def _load_merge_history(self):
        with open(self.MERGE_HISTORY_FILE_NAME, 'rb') as f:
            config.PrintDebug("Loading Merge History From {}".format(self.MERGE_HISTORY_FILE_NAME))
            self.merge_history = np.load(f)
        with open(self.MERGE_CLUSTER_HISTORY_FILE_NAME, 'rb') as f:
            config.PrintDebug("Loading Merge Cluster History From {}".format(self.MERGE_CLUSTER_HISTORY_FILE_NAME))
            self.cluster_history = pickle.load(f)
        pass
    
    ##########################################################
    # CLUSTER ASSIGNMENT
    def get_empty_clusterset_index(self, clusters):
        for i in range(len(clusters)):
            if len(clusters[i]) == 0:
                return i
        return -1
    
    def _create_clusters_from_training_data(self, num_clusters):
        # if recreating the clusters, delete the existing info
        if os.path.isdir(self.CLUSTER_INFO_FOLDER_NAME):
            shutil.rmtree(self.CLUSTER_INFO_FOLDER_NAME)
        
        # 1. Create array, one set for each cluster
        self.clusters = []
        for i in range(num_clusters):
            self.clusters.append(set())
        
        # 2. populate the first element with all elements
        for val in self.cluster_history[-1]:
            self.clusters[0].add(val)
        
        print("First has: {} elements".format(len(self.clusters[0])))
        # 3. loop and split of merged parts
        hist_ind = -2
        cluster_to_set = self.get_empty_clusterset_index(self.clusters)
        while cluster_to_set != -1:
            cluster_to_split = -1
            for val in self.cluster_history[hist_ind]:
                # find the right cluster to split
                if cluster_to_split == -1:
                    for k in range(num_clusters):
                        if val in self.clusters[k]:
                            cluster_to_split = k
                            break
                self.clusters[cluster_to_split].remove(val)
                self.clusters[cluster_to_set].add(val)
            hist_ind -= 1
            cluster_to_set = self.get_empty_clusterset_index(self.clusters)
 
    
        self.clusters = np.array(self.clusters)
        with open(self.CLUSTER_GROUPS_FILE_NAME, 'wb') as f:
            config.PrintDebug("Saving Cluster Groups Into {}".format(self.CLUSTER_GROUPS_FILE_NAME))
            np.save(f, self.clusters)
        
        if not os.path.exists(self.CLUSTER_INFO_FOLDER_NAME):
            os.makedirs(self.CLUSTER_INFO_FOLDER_NAME)
        
        # Save the image file names of each cluster into a file for evaluation
        for c in range(self.clusters.shape[0]):
            cluster_indices = list(self.clusters[c])
            with open("{}/Cluster_{}.txt".format(self.CLUSTER_INFO_FOLDER_NAME, c), 'w') as f:
                for cluster_index in cluster_indices:
                    f.write("{}\n".format(self.training_filenames[cluster_index]))
                    
        pass
    
    def _load_cluster_groups(self):
        with open(self.CLUSTER_GROUPS_FILE_NAME, 'rb') as f:
            config.PrintDebug("Loading Cluster Groups From {}".format(self.CLUSTER_GROUPS_FILE_NAME))
            self.clusters = np.load(f, allow_pickle=True)
        pass
    def _load_within_cluster_variance(self):
        with open(self.CLUSTER_VARIANCE_FILE_NAME, 'rb') as f:
            config.PrintDebug("Loading Cluster Variance From {}".format(self.CLUSTER_VARIANCE_FILE_NAME))
            self.within_cluster_variance = np.load(f, allow_pickle=True)
        pass
    
    def calculate_within_cluster_distance(self):
        print("Calculating Within-Cluster Distances")
        self.within_cluster_distances = [0.0] * len(self.clusters)
        
        for cluster_index in range(len(self.clusters)):
            # Calculate average
            cluster = list(self.clusters[cluster_index])
            total_dist = []
            for i in range(0, len(cluster)-1):
                for k in range(i+1, len(cluster)):
                    total_dist.append(self.proximity_matrix[cluster[i]][cluster[k]])
                    
            total_dist = np.nanmean(total_dist)
            self.within_cluster_distances[cluster_index] = total_dist
            
            # Calculate largest or largest minus average
#            largest_dist = 0.0 
#            for i in range(0, len(cluster)-1):
#                for k in range(i+1, len(cluster)):
#                    dist = self.proximity_matrix[cluster[i]][cluster[k]]
#                    if dist > largest_dist:
#                        largest_dist = dist
#            self.within_cluster_distances[cluster_index] = largest_dist - total_dist
            
            # Calculate minimum
#            smallest_dist = float('inf')
#            for i in range(0, len(cluster)-1):
#                for k in range(i+1, len(cluster)):
#                    dist = self.proximity_matrix[cluster[i]][cluster[k]]
#                    if dist < smallest_dist:
#                        smallest_dist = dist
#            self.within_cluster_distances[cluster_index] = smallest_dist
            
            print("Cluster {} has within-cluster distance of: {}".format(cluster_index, self.within_cluster_distances[cluster_index]))
        pass
    
    def calculate_within_cluster_variance(self):
        self.within_cluster_variance = [0.0] * len(self.clusters)
        for cluster_index in range(len(self.clusters)):
            print("Calculating Within-Cluster Variance for cluster: {}".format(cluster_index))
            cluster = list(self.clusters[cluster_index])
            
            within_dist_with = 0.0
            total_dist = []
            for i in range(0, len(cluster)-1):
                for k in range(i+1, len(cluster)):
                    total_dist.append(self.proximity_matrix[cluster[i]][cluster[k]])  
            within_dist_with = np.nanmean(total_dist)
            
            num_processes = config.MAX_AVAILABLE_CPU_CORES
            # Spread the work out around all the processes
            # Each process will have around the same number of data instances to handle, but higher index value has less calculations to do then lower index value data
            range_indices = np.array(range(0, len(cluster)))
            divided_indices = np.array_split(range_indices, num_processes)
            
            largest_diff = 0.0
            # Execute multi-processor functions
            with concurrent.futures.ProcessPoolExecutor(max_workers=config.MAX_AVAILABLE_CPU_CORES) as executor:
                results = [executor.submit(within_cluster_variance_calculator_thread, index, divided_indices[index][0], divided_indices[index][-1], cluster, self.proximity_matrix, within_dist_with) for index in range(num_processes)]   
            
                # collect the results as the processes complete and fill out the proximity matrix
                for f in concurrent.futures.as_completed(results):
                    result = f.result()
                    if result > largest_diff:
                        largest_diff = result
                
            self.within_cluster_variance[cluster_index] = largest_diff
            print("Cluster {}'s within-cluster distance variance is {}".format(cluster_index, largest_diff))
            
        with open(self.CLUSTER_VARIANCE_FILE_NAME, 'wb') as f:
            config.PrintDebug("Saving Cluster Variance Into {}".format(self.CLUSTER_VARIANCE_FILE_NAME))
            np.save(f, self.within_cluster_variance)
        pass
    
    ##########################################################
    # EVALUATION
    def analyse_generated_clusters(self):
        self.analyse_cluster_distances_to_other_clusters()
        self.analyse_clusters_using_original_data()
        pass
    
    def analyse_cluster_distances_to_other_clusters(self):
        print("Analysing internal cluster distances:")
        dists = []
        num_clusters = len(self.clusters)
        for c_one in range(num_clusters-1):
            for c_two in range(c_one+1, num_clusters):
                c_dist = self._get_cluster_distance_for_linkage(LINKAGE.AVERAGE, self.proximity_matrix, self.clusters[c_one], self.clusters[c_two])
                dists.append(c_dist)
                print("\tCluster {} vs Cluster {} === {}".format(c_one, c_two, c_dist))
        print("AVERAGE DIST: {}".format(np.average(dists)))
        pass
    def analyse_clusters_using_original_data(self):
        all_std_vals = []
        all_min_vals = []
        all_max_vals = []
        for cluster_index in range(len(self.clusters)):
            cluster = self.clusters[cluster_index]
            std_val = []
            min_val = []
            max_val = []
            for c_i in cluster:
                file_name = self.training_filenames[c_i]
                info_file_name = "{}/{}/{}/{}.txt".format(config.BASE_DATA_PATH, config.RESULT_FOLDER, self.SUB_FOLDER_PATH, file_name[:-4])
                with open(info_file_name, 'r') as file:
                    dict_info = json.loads(file.read())
                    std_val.append(dict_info['Original_std_val'])
                    min_val.append(dict_info['Original_min_val'])
                    max_val.append(dict_info['Original_max_val'])
            all_std_vals.append(std_val)
            all_min_vals.append(min_val)
            all_max_vals.append(max_val)
        
        print("STD:")
        for i in range(len(self.clusters)):
            print("Cluster {}\tstd: {:.4f} \t| min: {:.4f} \t| max: {:.4f} \t| mean: {:.4f}".format(i, np.std(all_std_vals[i]), np.amin(all_std_vals[i]), np.amax(all_std_vals[i]), np.average(all_std_vals[i])))
            
        print("MIN:")
        for i in range(len(self.clusters)):
            print("Cluster {}\tstd: {:.4f} \t| min: {:.4f} \t| max: {:.4f} \t| mean: {:.4f}".format(i, np.std(all_min_vals[i]), np.amin(all_min_vals[i]), np.amax(all_min_vals[i]), np.average(all_min_vals[i])))
        
        print("MAX:")
        for i in range(len(self.clusters)):
            print("Cluster {}\tstd: {:.4f} \t| min: {:.4f} \t| max: {:.4f} \t| mean: {:.4f}".format(i, np.std(all_max_vals[i]), np.amin(all_max_vals[i]), np.amax(all_max_vals[i]), np.average(all_max_vals[i])))
        pass
    
    ##########################################################
    # PREDICTION
    def _calculate_distance_to_cluster(self, cluster_index, predict_histogram, linkage_type):
        proxy_matrix_dupe = np.array(self.proximity_matrix, copy=True) # create duplicate of the initial proximity matrix
        size_of_proxy_matrix = proxy_matrix_dupe.shape[0]
        distances = []
        for p in range(size_of_proxy_matrix):
            hist_of_training = self.training_data[p]
            w_dist = wasserstein_distance(hist_of_training, predict_histogram)
            distances.append(w_dist)
        
        # Add the distances to the proximity matrix, both as column and as row
        proxy_matrix_dupe = np.vstack((proxy_matrix_dupe, distances)) # add new row to 2d matrix
        distances = np.append(distances, np.NaN) # add an extra NaN to the end of the list, this will be for the distance to itself
        proxy_matrix_dupe = np.column_stack((proxy_matrix_dupe, distances)) # add new column to 2d matrix
        
        dist_to_cluster = self._get_cluster_distance_for_linkage(linkage_type, proxy_matrix_dupe, self.clusters[cluster_index], [size_of_proxy_matrix])
        
        # subtract the average within-cluster distance
        dist_to_cluster -= self.within_cluster_distances[cluster_index]
        return dist_to_cluster
        
    def _calculate_distance_to_cluster_V2(self, cluster_index, predict_histogram):
        proxy_matrix_dupe = np.array(self.proximity_matrix, copy=True) # create duplicate of the initial proximity matrix
        size_of_proxy_matrix = proxy_matrix_dupe.shape[0]
        
        distances = []
        for p in range(size_of_proxy_matrix):
            hist_of_training = self.training_data[p]
            w_dist = wasserstein_distance(hist_of_training, predict_histogram)
            distances.append(w_dist)
            
        within_dist_before = 0.0
        total_dist = []
        cluster_dupe = list(self.clusters[cluster_index])
        for i in range(0, len(cluster_dupe)-1):
            for k in range(i+1, len(cluster_dupe)):
                total_dist.append(proxy_matrix_dupe[cluster_dupe[i]][cluster_dupe[k]])  
        within_dist_before = np.nanmean(total_dist)
        
        # Add the distances to the proximity matrix, both as column and as row
        proxy_matrix_dupe = np.vstack((proxy_matrix_dupe, distances)) # add new row to 2d matrix
        distances = np.append(distances, np.NaN) # add an extra NaN to the end of the list, this will be for the distance to itself
        proxy_matrix_dupe = np.column_stack((proxy_matrix_dupe, distances)) # add new column to 2d matrix
        
        #print(cluster_dupe)
        cluster_dupe.append(size_of_proxy_matrix)
        within_dist_after = 0.0
        total_dist = []
        for i in range(0, len(cluster_dupe)-1):
            for k in range(i+1, len(cluster_dupe)):
                total_dist.append(proxy_matrix_dupe[cluster_dupe[i]][cluster_dupe[k]])  
        within_dist_after = np.nanmean(total_dist)
        
        return within_dist_after - within_dist_before
    
    def _predict_cluster_and_accuracy(self, image_path, num_bins, num_clusters):
        # 1. Load image: load each image file and change the pixel range to be from 0 - 1.
        config.PrintDebug("Predicting: {}".format(image_path.split('\\')[-1]))
        
        image = skimage.io.imread(fname=image_path, as_gray=True) / config.U16_MAX_VAL
        # 2. Convert to histogram
        histogram, bin_edges = np.histogram(image, bins=num_bins, range=(0, 1))
        
        # 3. Find the right cluster and the distance to it
        cluster_distances = []
        for c in range(num_clusters):
            #dist_to_cluster = self._calculate_distance_to_cluster_V2(c, histogram)
            dist_to_cluster = self._calculate_distance_to_cluster(c, histogram, LINKAGE.AVERAGE)
            cluster_distances.append((c, dist_to_cluster))
        
        cluster_distances = sorted(cluster_distances, key = lambda x: x[1])  
        
        # 4. Subtrack the within cluster distance
        closest_cluster = cluster_distances[0][0]
        dist_to_closest = cluster_distances[0][1]        
        
        # if distance is less then 0 than it's inside the cluster
#        if dist_to_closest < 0:
#            dist_to_closest = 0
        
        # 5. Return results
        return (closest_cluster, dist_to_closest)
    
    ##########################################################

    pass # EoC

def intTryParse(text_val):
    try:
        return int(text_val), True
    except ValueError:
        return text_val, False


if __name__ == '__main__':
    NUM_BINS_FOR_HISTOGRAM = 512
    LINKAGE_TYPE = LINKAGE.AVERAGE
    SUB_FOLDER_PATH = "rand_samples_of_all_areas_512x512"
    
    force_rebuild_histogram        = False
    force_rebuild_init_proxy       = False
    force_rebuild_linkage          = False
    force_rebuild_cluster_grouping = False
    
    
    distance_threshold = 7000
    number_of_clusters = 32
    
    # Initialise class
    my_trsm_model = My_Terrain_Realism_Scoring_Metric_Class(NUM_BINS_FOR_HISTOGRAM, LINKAGE_TYPE)
    # Step 1: Turn images into histograms
    my_trsm_model.generate_histogram_data(SUB_FOLDER_PATH, force_rebuild_histogram)
    # Step 2: Create initial proximity matrix
    my_trsm_model.generate_proximity_matrix(force_rebuild_init_proxy)
    # Step 3: Merge all clusters to create merge hierarchy
    my_trsm_model.generate_merge_history_hierarchy(force_rebuild_linkage)
    
    
    # Get the user to select the right threshold value if it's not already set
    if distance_threshold == -1:
        my_trsm_model.display_dendrogram(0)
        while True:
            threshold_text = input("Enter threshold value. (q to quit): ")
            if threshold_text == 'q':
                break
            
            input_val, success = intTryParse(threshold_text)
            if success:
                distance_threshold = input_val
                my_trsm_model.display_dendrogram(distance_threshold)
#    else:
#        my_trsm_model.display_dendrogram(distance_threshold)


    while number_of_clusters == -1:
        num_cluster_text = input("Enter number of clusters: ")        
        input_val, success = intTryParse(num_cluster_text)
        if success:
            number_of_clusters = input_val

    if number_of_clusters > 0:
        
        my_trsm_model.generate_clusters_for_prediction(number_of_clusters, force_rebuild_cluster_grouping)
        my_trsm_model.analyse_generated_clusters()
               
        # TESTING HEIGHT MAPS USED BY SURVEY
        print("TESTING AGAINST SURVEY")
        test_glob_path = "{}\Survey_Images\SurveyHeightMaps\missed\*.png".format(config.BASE_DATA_PATH)
        image_paths_list = glob.glob(test_glob_path)
        results = []
        for file_path in image_paths_list:
            result_test = my_trsm_model.predict(file_path)
            results.append(result_test[1])
            print("{} belongs to cluster: {} with score of: {}".format(file_path[:-4], result_test[0], result_test[1]))
        print("Test completed. Max value: {}, Min value: {}, Mean: {}, STD: {}".format(np.amax(results), np.amin(results), np.average(results), np.std(results)))
        
        
        # TESTING AGAINST REAL IMAGES SEPARATED FROM THE DATA AFTER TRAINING
        print("TESTING AGAINST SUBSET")
        test_glob_path = "{}\{}\{}_subset\*.png".format(config.BASE_DATA_PATH, config.RESULT_FOLDER, SUB_FOLDER_PATH)
        image_paths_list = glob.glob(test_glob_path)
        results = []
        for file_path in image_paths_list:
            result_test = my_trsm_model.predict(file_path)
            results.append(result_test[1])
            print("{} belongs to cluster: {} with score of: {}".format(file_path[:-4], result_test[0], result_test[1]))
        print("Test completed. Max value: {}, Min value: {}, Mean: {}, STD: {}".format(np.amax(results), np.amin(results), np.average(results), np.std(results)))


        # TESTING GENERATED HEIGHT_MAPS PROVIDED BY RYAN S.
        print("TESTING AGAINST DATA BY RYAN S")
        test_glob_path = "Generated_Samples\ByRyanS\edited\*.png"
        image_paths_list = glob.glob(test_glob_path)
        results = []
        for file_path in image_paths_list:
            result_test = my_trsm_model.predict(file_path)
            results.append(result_test[1])
            print("{} belongs to cluster: {} with score of: {}".format(file_path[:-4], result_test[0], result_test[1]))
        print("Test completed. Max value: {}, Min value: {}, Mean: {}, STD: {}".format(np.amax(results), np.amin(results), np.average(results), np.std(results)))

        # TESTING AGAINST REAL IMAGES SEPARATED FROM THE DATA BEFORE TRAINING
        print("TESTING AGAINST TEST SET")
        test_glob_path = "{}\{}\{}_test\srtm_[0-9][0-9]_[0-9][0-9]_[0-9].png".format(config.BASE_DATA_PATH, config.RESULT_FOLDER, SUB_FOLDER_PATH)
        image_paths_list = glob.glob(test_glob_path)
        results = []
        for file_path in image_paths_list:
            result_test = my_trsm_model.predict(file_path)
            results.append(result_test[1])
            print("{} belongs to cluster: {} with score of: {}".format(file_path[:-4], result_test[0], result_test[1]))
        print("Test completed. Max value: {}, Min value: {}, Mean: {}, STD: {}".format(np.amax(results), np.amin(results), np.average(results), np.std(results)))
        
        # TESTING AGAINST PERLIN NOISE GENERATED IMAGES
        print("TESTING AGAINST PERLIN NOISE SET")
        test_glob_path = "{}\{}\perlin_generated\*.png".format(config.BASE_DATA_PATH, config.RESULT_FOLDER)
        image_paths_list = glob.glob(test_glob_path)
        results = []
        for file_path in image_paths_list:
            result_test = my_trsm_model.predict(file_path)
            results.append(result_test[1])
            print("{} belongs to cluster: {} with score of: {}".format(file_path[:-4], result_test[0], result_test[1]))
        print("Test completed. Max value: {}, Min value: {}, Mean: {}, STD: {}".format(np.amax(results), np.amin(results), np.average(results), np.std(results)))
