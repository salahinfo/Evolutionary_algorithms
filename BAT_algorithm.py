import numpy as np
import random
import math
import copy
import Graphtools 
import networkx as nx 
import networkx.algorithms.community as nx_comm
import time 
from sklearn.metrics.cluster import normalized_mutual_info_score
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

    def frequency(self):
        beta = random.uniform(0,1)
        fr_bat = self.f_min + (self.f_max - self.f_min)*beta
        
        return fr_bat

    def initialize_bat(self):
        pop_s = []
        PVelocity = []  
        for j in range(self.size):
            Velocity = {i : random.randint(0,1) for i in self.list_node}
            position = {i : None for i in self.list_node}
            com_id = 0

            for i in self.list_node:
                position.update({i : random.randint(0,self.size)})
            
            for k in position:
                for j in position:
                    if position[k] == position[j] and k not in self.G[j]:
                        val = random.choice(list(self.G[j]))
                        kv = position[val]
                        position.update({j : kv})                       
                 
            
                       
            pop_s.append(position)
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
            Q_value.append(self.Modularity(self.G, Pop_s[i]))

        index_best = Q_value.index(max(Q_value))    

        return Q_value , index_best
    
    def Update_velocity(self, Pop_s,Best_solution, velocity):
    

        for index in Pop_s :
            if Pop_s[index] == Best_solution[index]:
                X = 0
            else :
                X = 1
        
            Velo = velocity[index] + X * self.frequency() 
            sig = 1/(1+np.exp(-Velo))
            #print(sig)
            rn = random.uniform(0,1)
            if rn >= sig :
                velocity.update({index : 0})
            else :
                velocity.update({index : 1})

        return velocity 

    def tarans (self, clust_label):
        clust = np.array(list(clust_label.values()))
        clust = np.unique(clust)
        clusters = []
        for i in clust:
            cluste = []
            for j in clust_label:
                if clust_label[j] == i:   
                    cluste.append(j)
            
            clusters.append(cluste)

        return clusters 

    def Connect (self,vertex, cluste):
        edge = Graphtools.select_edge_betw(self.G,vertex,cluste)

        return edge

        

    def Update_position(self, Pop_s,velocity, rp):

        clusters = self.tarans(Pop_s) 
        #print("velocity ", velocity)  
        #print("befor update" , clusters)
        for index in Pop_s.keys() :
            
            rn = random.uniform(0,1)
            if velocity[index] == 1 and rn < rp :
        
                pos = -1
                maxc = 0
                for inde,cluster in enumerate(clusters):
                    if index in cluster:
                        pos = inde
                    else :    
                        btw = Graphtools.select_edge_betw(self.G, index, cluster)
                        
                        if btw > maxc :
                            maxc = btw
                            index_max = inde

                if maxc > 0:
                    clusters[pos].remove(index)
                    clusters[index_max].append(index)

        #print("after update",clusters)    
        membership = {i : None for i in self.list_node}
        for i in range(len(clusters)):
            for ind in clusters[i]:
                membership[ind] = i


        return  membership     

    def minvertex (self,cluster):
        edg_btw = dict()
        for i in cluster:
             edg_btw.update({ i : Graphtools.select_edge_betw(self.G,i,cluster)})  
            
        v_min = Graphtools.Min_dict(edg_btw)
        return v_min     

    def  local_solution (self, pop_s):
        edg_btw = dict()
        clusters = self.tarans(pop_s)
        listm =[]
        for clus in clusters:
            vertex = self.minvertex(clus)
            listm.append(vertex)

        for v_min in listm :
            maxc = 0
            for index,cluster in enumerate(clusters):
                if v_min in cluster:
                    pos = index
                else :    
                    btw = Graphtools.select_edge_betw(self.G, v_min, cluster)
                        
                    if btw > maxc :
                        maxc = btw
                        index_max = index

            if maxc > 0:
                clusters[pos].remove(v_min)
                clusters[index_max].append(v_min) 
        
         
        membership = {i : None for i in self.list_node}
        for i in range(len(clusters)):
            for index in clusters[i]:
                membership[index] = i


        return clusters , membership        

    def gene_sol(self,pop_s):
        clusters = self.tarans(pop_s)  
        #list_node = sorted([i for i in self.G.nodes()])              
        for index in pop_s.keys():
            pos = -1
            maxc = 0
            for inde,cluster in enumerate(clusters):
                if index in cluster:
                    pos = inde
                else :    
                    btw = Graphtools.select_edge_betw(self.G, index, cluster)
                        
                    if btw > maxc :
                        maxc = btw
                        index_max = inde

            if maxc > 0:
                clusters[pos].remove(index)
                clusters[index_max].append(index)

        #print("after update",clusters)    
        membership = {i : None for i in self.list_node}
        for i in range(len(clusters)):
            for ind in clusters[i]:
                membership[ind] = i

            
            #pop_s.append(position)
  
        return membership
 

    def Run_bat(self):

        start = time.time()
        position,velo_pos = self.initialize_bat()
        print(" after initilization ", position , " velocity ", velo_pos)
        Q_val , index_best = self.Fitness(position)
        print ( "q_val",Q_val," best pop", index_best)
        best_Qv = max(Q_val)
        rp =[]
        for i in range(len(position)):
            beta = random.uniform(0,1)        
            rp_bat = 0.1
            rp.append(0.1)

        Ab =[]
        for k in range(len(position)):    
            be = random.uniform(0,1)
            ab = be*(2-1)+1
            Ab.append(ab)

        #print(" pulse rate",rp)
        #print(" loudnees", Ab)
        best_pos = {}
        best_Q = []
        te = 0
        while te < self.num_iter:
            best_Qv = max(Q_val)    
            print(te, best_Qv)
            index_best = Q_val.index(best_Qv)
            for i in range(self.size):
                velo_pos[i] = self.Update_velocity(position[i], position[index_best], velo_pos[i])
                position[i] = self.Update_position(position[i],velo_pos[i],rp[i]) 
                Q_val[i] = nx_comm.modularity(self.G,self.tarans(position[i]))
   
                if random.uniform(0,1) >= rp[i]:
                  
                    cluster,pos_local = self.local_solution(position[i])
                    q_local = nx_comm.modularity(self.G,self.tarans(pos_local))

                    if q_local > Q_val[i]:
                        position[i] = pos_local.copy() 
                        Q_val[i] = q_local 
                        #if q_local > best_Qv:
                            #position[index_best] = pos_local.copy()
                            #Q_val[index_best] = q_local
                            
                        #print("after local position", position[i])
                r_pos = random.randint(0,self.size-1)
                new_sol = self.gene_sol(position[r_pos])
                new_sol_m = nx_comm.modularity(self.G,self.tarans(new_sol))
                if new_sol_m > best_Qv and random.uniform(0,1) < Ab[i]:
                    position[index_best] = new_sol.copy()
                    Q_val[index_best] = new_sol_m
                    Ab[i] = 0.9 * Ab[i]
                    rp[i] = 0.1 * (1 - math.exp(-0.9*te))
                    #print("after replace new solution",position[i])
                    #print("pulse rate", Ab, rp)
                
            current = time.time()- start
            print("time and modularity",current , best_Qv)
            best_Q.append(best_Qv)
            best_pos.update({ best_Qv : position[index_best]})              
            te += 1 
          
        end = time.time()-start
        ind = max(list(best_pos.keys()))
        posit = best_pos[ind]
            #print("pulse rate", Ab, rp)    

        return max(best_Q), posit, end

    

qval = []
NMI_list = []
Tm = []
path = sys.argv[1]
nbr_iter = 0
while nbr_iter < int(sys.argv[2]):
    #graph = nx.read_edgelist('/home/yacine/Desktop/real_network/metabolic.t' ,nodetype = int)
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
print(" the stedy value of modularity ", Graphtools.stdev(qval))
print(" avg value of NMI",max(NMI_list))
print(" max value of the modularity Q ", max(qval))
print(" time", Graphtools.avg(Tm))








