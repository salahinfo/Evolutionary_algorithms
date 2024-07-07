import numpy as np
import networkx as nx 
import random
import networkx.algorithms.community as nx_comm
import copy
import math
import Graphtools 
from statistics import mode
from sklearn.metrics.cluster import normalized_mutual_info_score
import time
import sys




class Batalgo :
    def __init__(self ,G , num_iter,f_max,f_min,r):
        self.G = G
        self.num_iter = num_iter
        self.f_max = f_max
        self.f_min = f_min
        self.r = r
        self.N = G.number_of_nodes()
        self.size = 100
        self.list_node = sorted([i for i in self.G.nodes()])
        print(self.list_node)

    def frequency(self):
        beta = random.uniform(0,1)
        fr_bat = self.f_min + (self.f_max - self.f_min)*beta
        
        return fr_bat

    def initialize_bat(self):
        pop_s = []
        PVelocity = []  
        for p in range(self.size):
            Velocity = {i : 0 for i in self.list_node}
            position = {i : random.randint(0,self.size) for i in self.list_node}
            com_id = 0
            
            alpha = 0.4
            t = round(alpha * self.N)
            index = random.sample(list(position.keys()), int(t))
            for k in index:
                comm_id = position[k]
                nghIndex = [nbr for nbr in self.G[k]]
                for ngh in nghIndex:
                    position.update({ngh : comm_id})
                   
                 
            
            position_n = Graphtools.normalize(position)               
            pop_s.append(position_n)
            PVelocity.append(Velocity)
        
        

        return pop_s, PVelocity 

    def Modularity(self, G, label):
        label = list(label.values())
        m = G.number_of_edges()
        node_list = sorted([i for i in G.nodes()])
        Q = 0
        for index,u in enumerate(node_list):
            if label[index] == None:
                continue
            for ind,v in enumerate(node_list):
                if label[ind] == None:
                    continue
                
                if v in list(nx.all_neighbors(G,u)):
                    Auv = 1
                else:
                    Auv = 0
                
                if  label[index] == label[ind] :
                    Q +=  1/(2*m) * (Auv - G.degree(u)*G.degree(v) / (2*m))
    
        return Q

    def Fitness (self , Pop_s):
        Q_value = []
        for i in range(len(Pop_s)):
            clust  = self.tarans(Pop_s[i])
            Q_value.append(nx_comm.modularity(self.G,clust))

        index_best = Q_value.index(max(Q_value))    

        return Q_value , index_best
    
    def Update_velocity(self, Pop_s, Best_solution, worst_solution, velocity, frq):
        
        ins_b = Graphtools.inters_bw(Pop_s,Best_solution)
        ins_w = Graphtools.inters_bw(Pop_s,worst_solution)
        Dth = dict()
        for i in ins_b :
            Dth.update({i : (ins_b[i]+ins_w[i])*frq +velocity[i]})
        
      
        for dth in Dth :
            if Dth[dth] >= 1:
                Dth.update({ dth : 1})
            else :
                Dth.update({ dth : 0})        

        return Dth
     
    
    def tarans (self, clust_label):
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

    def Connect (self,vertex, cluste):
        edge = Graphtools.select_edge_betw(self.G,vertex,cluste)

        return edge

    def Update_com (self , Pop_s, j):
        cont = []
        if  len(list(self.G.neighbors(j))) != 0:
            for ne in self.G.neighbors(j):
                cont.append(Pop_s[ne])  
        
            id_com = mode(cont)
        else :
            return Pop_s[j]    
        return id_com

    def Update_position(self, Pop_s, velocity):
        for j in Pop_s.keys():
            if velocity[j] == 1:
                id_c = self.Update_com(Pop_s,j)
                Pop_s.update({j : id_c})


        return  Pop_s     

    def edg_betw (self,cluster):
        edg_btw = dict()
        #edg_btw.update({ i : Graphtools.select_edge_betw(self.G,i,cluster)})  
        v_min = len(edg_btw)
        return v_min

    def  local_solution (self, pop_s):
        edg_btw = dict()
        clusters = self.tarans(pop_s)
        for vrtx in pop_s :
            maxc = 0
            for index,cluster in enumerate(clusters):
                if vrtx in cluster:
                    pos = index
                else :    
                    btw = Graphtools.select_edge_betw(self.G, vrtx, cluster)
                        
                    if btw > maxc :
                        maxc = btw
                        index_max = index

            if maxc > 0:
                clusters[pos].remove(vrtx)
                clusters[index_max].add(vrtx) 
        
         
        membership = {i : None for i in self.list_node}
        for i in range(len(clusters)):
            for index in clusters[i]:
                membership[index] = i

        label = Graphtools.normalize(membership)
        return clusters , label        


    

    def gene_sol(self , Pop_s):
        for j in Pop_s.keys():
           id_c = self.Update_com(Pop_s,j)
           Pop_s.update({j : id_c})
        solution = Graphtools.normalize(Pop_s)
        
        return  solution     

    def Run_bat(self):
        start = time.time()
        position,velo_pos = self.initialize_bat()
        #print(" after initilization ", position[0])
        Q_val , index_best = self.Fitness(position)
        q = max(Q_val)
        print ( "q_val", q," time", time.time()- start)
        frq =[]
        for i in range(len(position)):
            frq.append(self.frequency())
        R = 0.1
        A = 0.6
        #print(" pulse rate",R)
        #print(" loudnees", A)
        best_Q = []
        best_pos = {}
        te = 0
        while te < self.num_iter:
            best_Qv = max(Q_val)
            worst_qv = min(Q_val)    
            index_best = Q_val.index(best_Qv)
            index_worst = Q_val.index(worst_qv)
            for i in range(self.size):
                velo_pos[i] = self.Update_velocity(position[i], position[index_best], position[index_worst], velo_pos[i],frq[i])
                position[i] = self.Update_position(position[i],velo_pos[i]) 

                Q_val[i] = nx_comm.modularity(self.G,self.tarans(position[i]))
   
                if random.uniform(0,1) > R:
                  
                    cluster,pos_local = self.local_solution(position[index_best])
                    q_local = nx_comm.modularity(self.G,self.tarans(pos_local))
                    if q_local > best_Qv:
                        position[index_best] = pos_local.copy() 
                        Q_val[index_best] = q_local 
                      

                r_pos = random.randint(0,self.size-1)
                new_sol = self.gene_sol(position[r_pos])
                new_sol_m = nx_comm.modularity(self.G,self.tarans(new_sol))
                if new_sol_m > Q_val[i] and random.uniform(0,1) < A:
                    position[i] = new_sol.copy()
                    Q_val[i] = new_sol_m
                    A = 0.9 * A
                    R = 0.5 * (1 - math.exp(-0.9*te))
                   

            current = time.time()- start
            #print("time and modularity",current , best_Qv)
            best_Q.append(best_Qv)
            best_pos.update({ best_Qv : position[index_best]})              
            te += 1 
          
        end = time.time()-start
        ind = max(list(best_pos.keys()))
        posit = best_pos[ind]
        return max(best_Q) , posit, end

path = sys.argv[1]
qval = []
NMI_list = []
Tm = []
nbr_iter = 0    
while nbr_iter < int(sys.argv[2]):
    #graph = nx.read_edgelist('/home/yacine/Desktop/real_network/metabolic.t' ,nodetype = int)
    #graph = nx.read_gml( path, label = 'id')
    graph = Graphtools.Read_Graph(path)
    #graph = nx.read_edgelist('/home/yacine/Desktop/test.txt')
    g = Batalgo( graph, 100, 1, 0, 0.1)
    q, labelk , tm = g.Run_bat()
    if sys.argv[3] != 'None':
        True_partition = Graphtools.Read_GroundTruth(sys.argv[3])
        labelk = dict(sorted(labelk.items()))
        NMI = normalized_mutual_info_score(True_partition,  list(labelk.values()))
        NMI_list.append(NMI)

    qval.append(q)
    Tm.append(tm)
    nbr_iter += 1


avg = Graphtools.avg(qval)
print(" the stedy value of Q ", Graphtools.stdev(qval))
print(" The avearge modularity value",avg)
print(" The maximum modularity value", max(qval))
print("time", Graphtools.avg(Tm))
print(" The maximum NMI value", max(NMI))
