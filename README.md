# improved-influence-maximization
Methods to solve influence maximization problems.

1. static: IM methods in static social networks
2. dynamic: IM methods in dynamic social networks. Paper:

[1] Jabari Lotf, Jalil ,  M. Abdollahi Azgomi , and  M. R. Ebrahimi Dishabi . "An improved influence maximization method for social networks based on genetic algorithm." Physica A: Statistical Mechanics and its Applications 586(2022).

It is a method based on genetic algorithm in dynamic social networks, IC(Independent Cascade) diffusion model.

# dataset
1. CollegeMsg
2. email-Eu-core-temporal

from:
https://snap.stanford.edu/data/index.html


# result
On dataset CollegeMsg, set k = 5, t0 = 0, duration = 30, probability = 0.1

![image](https://github.com/chenyu-se/improved-influence-maximization/assets/17283947/d9dd29b8-8325-4cf2-b9ee-52f54dab1ffe)

['176', '1408', '281', '1097', '183'], the top k = 5 that influence the most nodes, up to 288 on average, in the dynamic network.


