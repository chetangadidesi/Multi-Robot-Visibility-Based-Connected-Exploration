# Multi-Robot-Visibility-Based-Connected-Exploration
Codebase for the ANTS 2025 Paper
(WIP -  Feel Free to Edit/ Add content where ever neccessary)

# Visibility
Initial efforts were done to calculate point visibility using the two CGAL Algorithms:
1) Rotational Sweep
2) Triangular Expansion

Point Visibility was then extended to Edge Visibility ( Visibiliy as seen by a line segment ). A heuristic was implemented in order to approximate the visibility region from a line segment as a Union of Point Visibility Polygons. This was done by discretizing/ sampling the line segment into multiple points and calculating the visibility polygons from each of these points and then combining the resulting visibility polygons.

( IDEA: Maybe a study can be done on how the number of samples affects the accuracy of the visibilty region. we can see how big of a difference does it make to have 10 samples vs a 100 and try to find an optimal sampling number)

Multiple simulations were performed to establish the fact the Triangular Expansion Algorithm was indeed faster than the Rotational Sweep for the use case that is currently of interest in this study. As per the origiginal CGAL paper (arXiv:1403.3905) the Rotational Sweep is an O (n log n) and the Triangular Expansion is O($n^2$) and hence for moderately complex geometries, the Triangular Expansion Algorithm is relatively faster.
The results from these simulations are illustrated below. 

1) Rotational Sweep
%image to be added here

2) Triangular Expansion
%image to be added here

The comparision of the run times is shown below.
%comparision table to be added here

Solving the 2-D visiblity problem is the first of many steps that were done in this research in order to meet our objective of Multi-Robot-Visibility-Based-Connected-Exploration.
After working on the problem of edge visibility then we moved on towards Min-Link Path Planning

# Min Link Path Planning
The idea of Min-Link Path Planning is that there exists a path between two points that has the least number of links/ turns. This would constraint us in moving only in straight lines and taking a turn in any direction would incur one additional cost. 
The goal is to minimize this cost while successfully moving from point A to point B. This problem could be solved by using the concept of a visibilty graph.
(WIP)



