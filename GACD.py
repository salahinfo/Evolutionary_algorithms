import numpy as np
import networkx as nx 
import random
import networkx.algorithms.community as nx_comm
import copy
import Graphtools
from sklearn.metrics.cluster import normalized_mutual_info_score
import sys 
import time 



class GACD:
    def __init__(self, G, size_pop, N_gens, Pc, Pm):
        self.G = G
        self.size_pop = size_pop
        self.Pc = Pc
        self.Pm = Pm
        self.N_gens = N_gens
      

    def initialization(self):
        node_list = sorted([i for i in self.G.nodes()])
        population = []
        for i in range(self.size_pop):
            Genotype = {}
            for j in node_list:
                n = random.choice(list(self.G.neighbors(j)))
                Genotype.update({j:n})
             
       
            population.append(Genotype)

        return population

    def decode (self , Genotype):
        label_clus = 0
        node_list = sorted([i for i in self.G.nodes()])
        clust_label = self.Convert(node_list)
        previous_ctr = self.Convert(node_list)
        for i in clust_label:
            ctr = 0
            if clust_label[i] == None:
               clust_label[i] = label_clus
               neighbour = Genotype[i]
               previous_ctr[ctr] = i
               ctr = ctr + 1
               while clust_label[neighbour] == None:
                    previous_ctr[ctr] = neighbour
                    clust_label[neighbour] = label_clus
                    neighbour = Genotype[neighbour]
                    ctr = ctr + 1
                
            if clust_label[neighbour] != label_clus:
                ctr = ctr-1
                while ctr >= 0 :
                    clust_label.update({ previous_ctr[ctr]:clust_label[neighbour]}) 
                    ctr = ctr - 1
            else: 
                    label_clus = label_clus + 1
           
        
        
        clust = np.array(clust_label)
        clust = np.unique(clust)
        clusters = []
        for i in clust:
            cluste = set()
            for index,j in enumerate(clust_label):
                if j == i:   
                    cluste.add(index)
            
            clusters.append(cluste)
       
        return clust_label            
    
    
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

    def Convert(self,lst):
        res_dct = { i: None for i in lst}
        return res_dct

    def Fillin(self ,pop_S):
        
        solution_p = dict()
        for index ,p in enumerate(pop_S) :
            pmm = self.decode(p)
            comm = Graphtools.tarans(pmm)
            solution_p.update({nx_comm.modularity(self.G,comm): p})
         
        return solution_p

                
    def Crossover(self,parent_1,parent_2):
        k = int(self.G.number_of_nodes()/2)
        ine_sa = random.sample(list(parent_1.keys()),k)
        ine_sb = random.sample(list(parent_2.keys()),k)
        for ina in range(len(ine_sa)): 

            val_sa = parent_1[ine_sa[ina]]
            val_sb = parent_2[ine_sb[ina]]
            
            parent_1.update({val_sa : parent_2[val_sa]})
            parent_1.update({ine_sb[ina]: parent_2[ine_sb[ina]]})
            
            
            parent_2.update({val_sb : parent_1[val_sb]})
            parent_2.update({ine_sa[ina] : parent_1[ine_sa[ina]]})
            
       
        return parent_1,parent_2   

    def Mutation (self, genome_1, genome_2):
        
        k = int(self.G.number_of_nodes()/2)
        index_genp = random.sample(list(genome_1.keys()), k)
        
        for ia in index_genp :
            ka = genome_1[ia]
            mutate_gene = random.choice(list(self.G.neighbors(ka)))
            genome_1.update({ia : mutate_gene})

        index_genpb = random.sample(list(genome_2.keys()),k)
        for ib in index_genpb :
            ka = genome_2[ib]
            mutate_geneb = random.choice(list(self.G.neighbors(ka)))
            genome_2.update({ib : mutate_geneb})

       
        return genome_1,genome_2
    
    def Update(self , population_s, curent_pop):

        population_s.update(curent_pop)
        population = sorted(list(population_s.keys()),reverse=True)
        population = population[    :self.size_pop]

        pop_s = { i : population_s[i] for i in population}

        return pop_s

    def Run_GA(self):
        start = time.time()
        Population = self.initialization()
        fitnes_solution = self.Fillin(Population)

        for g in range(self.N_gens):
            i = 0
            pp = dict()
            while i < self.size_pop :
                a = random.uniform(0, 1) 
                if a <= self.Pc:
                    pop_solution1,pop_solution2 = self.Crossover(Population[i],Population[i+1])
                    labl = self.decode(pop_solution1)
                    labl2 = self.decode(pop_solution2)
                    communities1 = Graphtools.tarans(labl)
                    communities2 = Graphtools.tarans(labl2)
                    pp.update({nx_comm.modularity(graph ,communities1):pop_solution1})
                    pp.update({nx_comm.modularity(graph ,communities2):pop_solution2})
                else:
                    pop_solution1,pop_solution2 = self.Mutation(Population[i],Population[i+1])
                    labl = self.decode(pop_solution1)
                    labl2 = self.decode(pop_solution2)
                    communities1 = Graphtools.tarans(labl)
                    communities2 = Graphtools.tarans(labl2)
                    pp.update({nx_comm.modularity(graph ,communities1):pop_solution1})
                    pp.update({nx_comm.modularity(graph ,communities2):pop_solution2})

                i = i+2
            fitnes = self.Update(pp , fitnes_solution)
            Population = copy.deepcopy(list(fitnes.values()))
            fitnes_solution = copy.deepcopy(fitnes)
            
        
        for Qu, community in fitnes.items():
            Q = Qu
            commu = community
            break
        
        end = time.time()-start
        return Q , commu, end 
           








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
    g = GACD(graph, 70, 114, 0.8, 0.2)

    q, labelk , tm = g.Run_GA()
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




