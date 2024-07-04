# Introduction 
This repositry provide an implemntation of evolutionary algorithm including BAT algorithm and Genetic algorithm to define the community structure in complex network.
1) GACD the paper titeled " A new genetic algorithm for community detection" [1], 
2) DBAT  "Community detection using discrete bat algorithm" [2]  
3) DBAT-M "Assessment of Discrete BAT-Modified (DBAT-M) Optimization Algorithm for Community Detection in Complex Network"[3] 

# Usage 
Note that the algorithm has been executed ten times to give you the metric value's maximum, average, and standard deviation. Install all requirements packages mentioned in the file requirments.txt and download the datasets you want to apply the algorithm. Then, execute the below command line if the networks have no ground truth. Execute this command.
f the networks have no ground truth. Execute this command.  
 ```
Genetic algorithm : python3  GACD.py  path_of_datasets  100  0.5  None  number of run algorihm
BAT algorithm : python3  DBA_M.py  path_of_datasets  100  0.5  None  number of run algorihm 

 ```
if the networks have ground-truth excute this commmnad 
```
Genetic algorithm : python3  GACD.py  path_of_datasets  100  0.5  None  number of run algorihm
BAT algorithm : python3  DBA_M.py  path_of_datasets  100  0.5  None  number of run algorihm 

```
# Reference :
[1] : https://link.springer.com/chapter/10.1007/978-3-642-02469-6_11#citeas
[2] : https://www.iaeng.org/IJCS/issues_v43/issue_1/IJCS_43_1_05.pdf
[3] : https://link.springer.com/article/10.1007/s13369-022-07229-y
