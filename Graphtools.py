import math
import re
import sys
import random
import numpy as np 
import networkx as nx





def select_edge_betw(g, v, cluster):
        Edg_betw = []
        nghIndex = [nbr for nbr in g[v]]
        for i in nghIndex:
            if i in cluster and i != v:
                Edg_betw.append(i)

        k = len(Edg_betw)        
    
        return k


def Min_dict (l):
	lst = list(l.values())
	lst_k = list(l.keys())
	k = lst_k[lst.index(min(lst))]
	return k

def tarans (clust_label):
    clust = np.array(list(clust_label.values()))
    clust = np.unique(clust)
    clusters = []
    for i in clust:
        cluste = set()
        for j in clust_label:
            if clust_label[j] == i:   
                cluste.add(j)
            
        clusters.append(cluste)

    return clusters
 
def N_list (pop_s,G,list_node):
        clusters = tarans(pop_s)
        ngh_sol = []
        for node in pop_s.keys():
            maxc = 0
            for index,cluster in enumerate(clusters):
                if node in cluster:
                    pos = index
                else :    
                    btw = select_edge_betw(G, node, cluster)
                        
                    if btw > maxc:
                        maxc = btw
                        index_max = index

            if maxc > 0:
                clusters[pos].remove(node)
                clusters[index_max].append(node) 
                membership = {i : None for i in list_node}
                for i in range(len(clusters)):
                    for index in clusters[i]:
                        membership[index] = i
                
                ngh_sol.append(membership)
                
        return ngh_sol


def inters_bw(ps,ps_2):
    insd = {i : 1 for i in ps}
    for ins in ps:
        if ps[ins] in ps_2.values():
            insd.update({ins : 0})
            
    return insd         

def normalize(values):
   
    val = values.values()
    uniques = list(dict.fromkeys(val))
    mapping = { value :uniques.index(values[value]) for value in values }
    return mapping

def Read_Graph(Path):
        
    if Path[len(Path)-3: ] == 'txt' or Path[len(Path)-3: ] == 'dat':
        Graph = nx.read_edgelist(Path, nodetype = int)
    elif Path[len(Path)-3: ] == 'gml':
        Graph = nx.read_gml(Path,label = 'id')
    else :
        raise TypeError (" the type of graph is not suportable or not no such file or directory")

    return Graph
    
def Reve(x):
    sa = x.split()[::-1]
    l = []
    for i in sa:
        l.append(i)

    l=('  '.join(l))
    return l

def Remove_Revers(path):
    with open(path, "r") as file:
        lines = file.readlines()
        result=[]
        for xa in lines:
            xa=re.sub(r'\s','  ',xa)
            if Reve(xa) not in result:
                result.append(xa.strip())   

    return(result)

def Remove_Dublicate (path):
    pathw = path[ : -3]+'txt' 
    lines = Remove_Revers(path)
    with open(pathw, 'w') as f:
        for line in lines:
            f.write(line)
            f.write('\n')


def Read_GroundTruth(path):
    with open(path, "r") as file:
        lines = file.readlines()
        result = []
        for x in lines:
            x = x.rstrip()
            result.append(x.split()[1])

    true_partion = [int(x)for x in result]
    return true_partion
    
   
def Is_Intersiction(communities):
    dupes = []
    flat = [item for sublist in communities for item in sublist]
    for f in flat:
        if flat.count(f) > 1:
            if f not in dupes:
                dupes.append(f)

    if dupes:

        print(" there is intersection") 
    else:
        print(" there is no intersection") 
         
        
def summ(arg):
    if len(arg) < 1:
        return None
    else:
        return sum(arg)

def count(arg):
    return len(arg)
  
def Min(arg):
    if len(arg) < 1:
        return None
    else:
        return min(arg)
  
def max(arg):
    if len(arg) < 1:
        return None
    else:
        return max(arg)
  
def avg(arg):
    if len(arg) < 1:
        return None
    else:
        return sum(arg) / len(arg)   
  
def median(arg):
    if len(arg) < 1:
        return None
    else:
        arg.sort()
        return  arg[len(arg) // 2]
  
def stdev(arg):
    if len(arg) < 1 or len(arg) == 1:
        return None
    else:
        avgG = avg(arg)
        sdsq = sum([(i - avgG) ** 2 for i in arg])
        stdev = (sdsq / (len(arg) - 1)) ** .5
        return stdev
  
def percentile(arg):
    if len(arg) < 1:
        value = None
    elif (arg >= 100):
        sys.stderr.write('ERROR: percentile must be < 100.  you supplied: %s\n'% arg)
        value = None
    else:
        element_idx = int(len(arg) * (arg / 100.0))
        arg.sort()
        value = arg[element_idx]
    return value  
    

def is_edge_betw(g,vert,commu):
    Edg_betw = []
    for i in list(g.neighbors(vert)):
        if i in commu :
            return True
            
    return False
    
def random_community(weights):
    x = random.random()
    for i in range(len(weights)):
        if x < weights[i]:
            return i

    
def weighted_choice(objects, weights):
    weights = np.array(weights, dtype=np.float64)
    sum_of_weights = weights.sum()
    np.multiply(weights, 1 / sum_of_weights, weights)
    weights = weights.cumsum()
    x = random.random()
    for i in range(len(weights)):
        if x < weights[i]:
            return objects[i]
        

    