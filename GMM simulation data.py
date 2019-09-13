#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 15:45:52 2019

@author: morleycoulson
"""

import numpy as np
import pandas as pd
from sklearn import cluster, datasets, mixture
from plotnine import *


if __name__ == '__main__':
    n_samples = 1500
    blobs = datasets.make_blobs(n_samples, cluster_std=[1.0, 2.5, 0.5])[0]
    #transformation = [[0.6, -0.6], [-0.4, 0.8]]
    #blobs = np.dot(blobs, transformation)

    gmm = mixture.GaussianMixture(n_components = 3, covariance_type = 'full')
    gmm.fit(blobs)
    
    kmeans = cluster.KMeans(n_clusters = 3)
    kmeans.fit(blobs)
    
    assignment = gmm.predict(blobs)
    assignment_kmeans = kmeans.predict(blobs)
    
    df_plot = pd.DataFrame(blobs, columns = ['x', 'y'])
    df_plot['gmm'] = assignment
    df_plot['kmeans'] = assignment_kmeans
    df_plot = pd.melt(df_plot, id_vars = ['x', 'y'], var_name = 'technique', value_name = 'assignment')
    df_plot['assignment'] = df_plot['assignment'].astype('object')
    
    plot = (ggplot(df_plot,
                   aes(x = 'x',
                       y = 'y',
                       colour = 'assignment')) +
            facet_wrap('~technique') +
            geom_point() +
            theme_bw())
    print(plot)