##  RESCUER AGENT
### @Author: Tacla (UTFPR)
### Demo of use of VictimSim
### Not a complete version of DFS; it comes back prematuraly
### to the base when it enters into a dead end position


import os
import random
import math
from map import Map
from vs.abstract_agent import AbstAgent
from vs.physical_agent import PhysAgent
from vs.constants import VS
from abc import ABC, abstractmethod
from utilities import Stack, Node
import itertools
import heapq

class Centroid:
    def __init__(self, posX, posY):
        self.posX = posX
        self.posY = posY

class Cluster:
    def __init__(self, posX, posY):
        self.centroid = Centroid(posX, posY)
        self.victims = {} # a dictionary of found victims: (seq): ((x,y), [<vs>])

## Classe que define o Agente Rescuer com um plano fixo
class Rescuer(AbstAgent):
    rescuers = []

    def __init__(self, env, config_file):
        """ 
        @param env: a reference to an instance of the environment class
        @param config_file: the absolute path to the agent's config file"""

        super().__init__(env, config_file)

        # Specific initialization for the rescuer
        self.map = None             # explorer will pass the map
        self.victims = None         # list of found victims
        self.plan = []              # a list of planned actions
        self.plan_x = 0             # the x position of the rescuer during the planning phase
        self.plan_y = 0             # the y position of the rescuer during the planning phase
        self.plan_visited = set()   # positions already planned to be visited 
        self.plan_rtime = self.TLIM # the remaing time during the planning phase
        self.plan_walk_time = 0.0   # previewed time to walk during rescue
        self.x = 0                  # the current x position of the rescuer when executing the plan
        self.y = 0                  # the current y position of the rescuer when executing the plan
        self.clusters = []    # list of clusters

        self.width = env.dic["GRID_WIDTH"]
        self.height = env.dic["GRID_HEIGHT"]
        self.baseX = env.dic["BASE"][0]
        self.baseY = env.dic["BASE"][1]
        self.base = Node(env.dic["BASE"][0], env.dic["BASE"][1])

        Rescuer.rescuers.append(self)
        # Starts in IDLE state.
        # It changes to ACTIVE when the map arrives
        self.set_state(VS.IDLE)
        self.path = Stack()

    def go_save_victims(self, map, victims):
        """ The explorer sends the map containing the walls and
        victims' location. The rescuer becomes ACTIVE. From now,
        the deliberate method is called by the environment"""

        print(f"\n\n*** R E S C U E R ***")
        self.map = map
        print(f"{self.NAME} Map received from the explorer")
        #self.map.draw()

        print('VITICMS: ')
        #print(f"{self.NAME} List of found victims received from the explorer")
        self.victims = victims
        for seq, ((x, y), vs) in self.victims.items():
            print(f"({seq}): (({x},{y}), {vs})")
        # print the found victims - you may comment out
        #for seq, data in self.victims.items():
        #    coord, vital_signals = data
        #    x, y = coord
        #    print(f"{self.NAME} Victim seq number: {seq} at ({x}, {y}) vs: {vital_signals}")

        #print(f"{self.NAME} time limit to rescue {self.plan_rtime}")

        self.__planner()
        print(f"{self.NAME} PLAN")
        i = 1
        self.plan_x = 0
        self.plan_y = 0
        for a in self.plan:
            self.plan_x += a[0]
            self.plan_y += a[1]
            print(f"{self.NAME} {i}) dxy=({a[0]}, {a[1]}) vic: a[2] => at({self.plan_x}, {self.plan_y})")
            i += 1

        print(f"{self.NAME} END OF PLAN")
                  
        self.set_state(VS.ACTIVE)
        
    def __depth_search(self, actions_res):
        enough_time = True
        ##print(f"\n{self.NAME} actions results: {actions_res}")
        for i, ar in enumerate(actions_res):

            if ar != VS.CLEAR:
                ##print(f"{self.NAME} {i} not clear")
                continue

            # planning the walk
            dx, dy = Rescuer.AC_INCR[i]  # get the increments for the possible action
            target_xy = (self.plan_x + dx, self.plan_y + dy)

            # checks if the explorer has not visited the target position
            if not self.map.in_map(target_xy):
                ##print(f"{self.NAME} target position not explored: {target_xy}")
                continue

            # checks if the target position is already planned to be visited 
            if (target_xy in self.plan_visited):
                ##print(f"{self.NAME} target position already visited: {target_xy}")
                continue

            # Now, the rescuer can plan to walk to the target position
            self.plan_x += dx
            self.plan_y += dy
            difficulty, vic_seq, next_actions_res = self.map.get((self.plan_x, self.plan_y))
            #print(f"{self.NAME}: planning to go to ({self.plan_x}, {self.plan_y})")

            if dx == 0 or dy == 0:
                step_cost = self.COST_LINE * difficulty
            else:
                step_cost = self.COST_DIAG * difficulty

            #print(f"{self.NAME}: difficulty {difficulty}, step cost {step_cost}")
            #print(f"{self.NAME}: accumulated walk time {self.plan_walk_time}, rtime {self.plan_rtime}")

            # check if there is enough remaining time to walk back to the base
            if self.plan_walk_time + step_cost > self.plan_rtime:
                enough_time = False
                #print(f"{self.NAME}: no enough time to go to ({self.plan_x}, {self.plan_y})")
            
            if enough_time:
                # the rescuer has time to go to the next position: update walk time and remaining time
                self.plan_walk_time += step_cost
                self.plan_rtime -= step_cost
                self.plan_visited.add((self.plan_x, self.plan_y))

                if vic_seq == VS.NO_VICTIM:
                    self.plan.append((dx, dy, False)) # walk only
                    #print(f"{self.NAME}: added to the plan, walk to ({self.plan_x}, {self.plan_y}, False)")

                if vic_seq != VS.NO_VICTIM:
                    # checks if there is enough remaining time to rescue the victim and come back to the base
                    if self.plan_rtime - self.COST_FIRST_AID < self.plan_walk_time:
                        print(f"{self.NAME}: no enough time to rescue the victim")
                        enough_time = False
                    else:
                        self.plan.append((dx, dy, True))
                        #print(f"{self.NAME}:added to the plan, walk to and rescue victim({self.plan_x}, {self.plan_y}, True)")
                        self.plan_rtime -= self.COST_FIRST_AID

            # let's see what the agent can do in the next position
            if enough_time:
                self.__depth_search(self.map.get((self.plan_x, self.plan_y))[2]) # actions results
            else:
                return

        return
    
    def __planner(self):
        """ A private method that calculates the walk actions in a OFF-LINE MANNER to rescue the
        victims. Further actions may be necessary and should be added in the
        deliberata method"""

        """ This plan starts at origin (0,0) and chooses the first of the possible actions in a clockwise manner starting at 12h.
        Then, if the next position was visited by the explorer, the rescuer goes to there. Otherwise, it picks the following possible action.
        For each planned action, the agent calculates the time will be consumed. When time to come back to the base arrives,
        it reverses the plan."""

        # This is a off-line trajectory plan, each element of the list is a pair dx, dy that do the agent walk in the x-axis and/or y-axis.
        # Besides, it has a flag indicating that a first-aid kit must be delivered when the move is completed.
        # For instance (0,1,True) means the agent walk to (x+0,y+1) and after walking, it leaves the kit.
        
        first_victim = next(iter(self.victims.items()))
        self.astar((0, 0), (first_victim[1][0]))

        while self.path.is_empty() == False:
            dx, dy = self.path.pop()
            self.plan.append((dx, dy, False)) # walk only 

        #print("Astar: de (" + str(self.x) + ", " + str(self.y) + ") a ...")
        #print(f"{self.NAME} {self.id} A* path: " + str(len(self.path.items)) + ' '.join(str(x) for x in self.path.items) )
            
        for seq, ((x, y), vs) in self.victims.items():
            print(f"({seq}): (({x},{y}), {vs})")      

        iterador_atual, iterador_proximo = itertools.tee(self.victims.items())
        next(iterador_proximo, None)  # Avança o segundo iterador para o segundo elemento

        # Itera e imprime o atual e o próximo
        for atual, proximo in zip(iterador_atual, iterador_proximo):
            self.astar(atual[1][0], proximo[1][0])
            while self.path.is_empty() == False:
                dx, dy = self.path.pop()
                self.plan.append((dx, dy, False)) # walk only 
            print(f"Atual: {atual}, Próximo: {proximo}")
            
        ultima_chave, ultimo_valor = list(self.victims.items())[-1]
        #ultimo_elemento = next(iterador_proximo, None)
        #if ultimo_elemento:
        #    print(f"Último elemento: {ultimo_elemento}")

        self.astar((ultimo_valor[0][0], ultimo_valor[0][1]), (0, 0))
        while self.path.is_empty() == False:
            dx, dy = self.path.pop()
            self.plan.append((dx, dy, False)) # walk only 

        '''
        aux = Stack()
        while self.plan.is_empty() == False:
            dx, dy = self.path.pop()
            aux.push((dx, dy))

        while aux.is_empty() == False:
            dx, dy = aux.pop()
            self.plan.append((dx, dy, False)) # walk only '''

        ''' old planner
        self.plan_visited.add((0,0)) # always start from the base, so it is already visited
        difficulty, vic_seq, actions_res = self.map.get((0,0))
        self.__depth_search(actions_res)

        # push actions into the plan to come back to the base
        if self.plan == []:
            return

        come_back_plan = []

        for a in reversed(self.plan):
            # triple: dx, dy, no victim - when coming back do not rescue any victim
            come_back_plan.append((a[0]*-1, a[1]*-1, False))

        self.plan = self.plan + come_back_plan
        '''

    def deliberate(self) -> bool:
        """ This is the choice of the next action. The simulator calls this
        method at each reasonning cycle if the agent is ACTIVE.
        Must be implemented in every agent
        @return True: there's one or more actions to do
        @return False: there's no more action to do """

        # No more actions to do
        if self.plan == []:  # empty list, no more actions to do
           #input(f"{self.NAME} has finished the plan [ENTER]")
           return False

        # Takes the first action of the plan (walk action) and removes it from the plan
        dx, dy, there_is_vict = self.plan.pop(0)
        #print(f"{self.NAME} pop dx: {dx} dy: {dy} vict: {there_is_vict}")

        # Walk - just one step per deliberation
        walked = self.walk(dx, dy)

        # Rescue the victim at the current position
        if walked == VS.EXECUTED:
            self.x += dx
            self.y += dy
            #print(f"{self.NAME} Walk ok - Rescuer at position ({self.x}, {self.y})")
            # check if there is a victim at the current position
            if there_is_vict:
                rescued = self.first_aid() # True when rescued
                if rescued:
                    print(f"{self.NAME} Victim rescued at ({self.x}, {self.y})")
                else:
                    print(f"{self.NAME} Plan fail - victim not found at ({self.x}, {self.x})")
        else:
            print(f"{self.NAME} Plan fail - walk error - agent at ({self.x}, {self.x})")
            
        #input(f"{self.NAME} remaining time: {self.get_rtime()} Tecle enter")

        return True

    def make_groups_victims(self, victims):
        max_X = self.width
        max_Y = self.height
        print(str(self.width) + str(self.height))

        for i in range(4):
            x = random.randint(1, max_X) - self.baseX
            y = random.randint(1, max_Y) - self.baseY
            print("posicao cluster: " + str(x) + " " + str(y))
            cluster = Cluster(x, y)
            self.clusters.append(cluster)

        changedClusterPosition = True
		
        iters = 1
        MAX_ITERATIONS = 500
        while changedClusterPosition and iters < MAX_ITERATIONS:
            changedClusterPosition = False
            for seq, ((x, y), vs) in victims.items():
                dist = float('inf')
                for j, elem in enumerate(self.clusters):
                    aux = math.sqrt(math.pow(x - elem.centroid.posX, 2) + math.pow(y - elem.centroid.posY, 2))
                    if aux < dist:
                        dist = aux
                        clusterIndex = j
                    if seq in elem.victims:
                        del elem.victims[seq]
                self.clusters[clusterIndex].victims[seq] = ((x, y), vs)

            for i in self.clusters:
                sumX = 0
                sumY = 0
                auxCentroid = Centroid(i.centroid.posX, i.centroid.posY)
                for seq, ((x, y), vs) in i.victims.items():
                    sumX = sumX + x
                    sumY = sumY + y

                print("tamanho dic victims: " + str(len(i.victims)))
                if(len(i.victims) != 0):                
                    i.centroid.posX = sumX/len(i.victims)
                    i.centroid.posY = sumY/len(i.victims)
                    if round(auxCentroid.posX, 1) != round(i.centroid.posX, 1) or round(auxCentroid.posY, 1) != round(i.centroid.posY, 1):
                        changedClusterPosition = True
            iters += 1

        for i, elem in enumerate(self.clusters):
            with open('cluster' + str(i) + '.txt', 'w') as arquivo:
                for seq, ((x, y), vs) in elem.victims.items():
                    arquivo.write(str(seq) + "," + str(x) + "," + str(y) + "," + str(vs[0]) + "," + str(vs[1]) + "," + str(vs[2]) + "," + str(vs[3]) + "," + str(vs[4]) + "," + str(vs[4]) + "\n")
                            
    def assign_groups_to_rescuers(self):
        clustersCopy = self.clusters
        random.shuffle(clustersCopy)

        for i in range(len(clustersCopy)):
            self.rescuers[i].cluster = clustersCopy[i]

    def receive_map_victims(self, map, victims):
        #self.assign_groups_to_rescuers()
        self.make_groups_victims(victims)
        
        self.sum_of_squared_error()     # SSE analysis

        self.silhouette_analysis()      # silhoutte analysis

        for i, rescuer in enumerate(Rescuer.rescuers):
            rescuer.go_save_victims(map, self.clusters[i].victims)

    def sum_of_squared_error(self):
        print("SSE analysis for the generated clusters")
        for i, elem in enumerate(self.clusters):
            SSE = 0   
            for seq, ((x, y), vs) in elem.victims.items():
                aux = math.sqrt(math.pow(x - elem.centroid.posX, 2) + math.pow(y - elem.centroid.posY, 2))
                SSE += aux
            print("SSE cluster" + str(i) + ": " + str(SSE))

    def silhouette_analysis(self):

        clustersAnalysis = []
        for i, elem in enumerate(self.clusters):
            if(len(elem.victims.items()) > 0):
                clustersAnalysis.append(elem)

        print("Silhouette analysis for the generated clusters")
        for i, elem in enumerate(clustersAnalysis):
            sumS = 0
            for seq, ((x, y), vs) in elem.victims.items():
                
                # intracluster
                sum = 0
                for seq1, ((x1, y1), vs1) in elem.victims.items():
                    if(seq != seq1):
                        aux = math.sqrt(math.pow(x - x1, 2) + math.pow(y - y1, 2))
                        sum += aux
                averageIntraCluster = sum / len(elem.victims.items())
                a = averageIntraCluster

                # interclusters
                averagesInterCluster = []
                for j, elem2 in enumerate(clustersAnalysis):
                    if(elem != elem2):
                        sum = 0
                        for seq1, ((x1, y1), vs1) in elem2.victims.items():
                            aux = math.sqrt(math.pow(x - x1, 2) + math.pow(y - y1, 2))
                            sum += aux
                        average = sum / len(elem2.victims.items())
                        averagesInterCluster.append(average)
                b = min(averagesInterCluster) if averagesInterCluster else float('inf')

                s = (b - a)/max(a, b)
                
                sumS += s

            averageS = sumS / len(elem.victims.items())
            print("Cluster" + str(i) + ": " + str(averageS))
            continue
    
    # COLOCAR ASTAR EM LUGAR QUE EXPLORER E RESCUER POSSAM USAR O MESMO

    def chebyshev(self, node, end_node):      # heuristica
        return max(abs(node.x - end_node.x), abs(node.y - end_node.y))

    def get_neighbors(self, node):
        neighbors = []        
        
        item = self.map.get((node.x, node.y))
        if(item):
            obstacules = item[2]
        else:
            obstacules = [1, 1, 1, 1, 1, 1, 1, 1,]

        #print(' '.join(str(x) for x in obstacules))
        for key, incr in Rescuer.AC_INCR.items():
            new_x, new_y = node.x + incr[0], node.y + incr[1]

            #print("new node: " + str(new_x) + " " + str(new_y))
            if obstacules[key] == 0 and new_x + self.base.x >= 0 and new_y + self.base.y >= 0 and new_x + self.base.x < self.width and new_y + self.base.y < self.height:
                #print("obstacules aceitos: " + str(obstacules[key]))
                neighbors.append(Node(new_x, new_y, node))
        return neighbors

    def astar(self, start, end):
        open_list = []
        closed_set = set()
        #print("start node: " + str(start[0]) + str(start[1]))
        start_node = Node(start[0], start[1])
        end_node = Node(end[0], end[1])
        heapq.heappush(open_list, start_node)
        
        while open_list:
            #current_node = next((obj for obj in open_list if obj.x == 0 and obj.y == 0), None)

            #if current_node is None:
            current_node = heapq.heappop(open_list)
            
            #print("current node: " + str(current_node.x) + " x " + str(current_node.y))
            if current_node.x == end_node.x and current_node.y == end_node.y:
                #path = []
                #print("TERMINANDO")
                while current_node:
                    if(current_node.parent is not None):
                        self.path.push((current_node.x - current_node.parent.x, current_node.y - current_node.parent.y))
                    else:
                        self.path.push((0,0))
                    current_node = current_node.parent
                #print(str(self.path.pop()))
                return
            
            closed_set.add((current_node.x, current_node.y))
            
            for neighbor in self.get_neighbors(current_node):
                aux = next((obj for obj in closed_set if obj[0] == neighbor.x and obj[1] == neighbor.y), None)
                if aux is not None:	# testar
                    continue
                
                #if neighbor in closed_set:
                    #continue
                
                g_score = current_node.g + 1
                h_score = self.chebyshev(neighbor, end_node)
                f_score = g_score + h_score
                
                #print("scores: " + str(g_score) + " " + str(h_score) + " " + str(f_score) )

                aux = next((obj for obj in open_list if obj.x == neighbor.x and obj.y == neighbor.y), None)

                if aux is not None:
                    if neighbor.g < aux.g:
                        aux.g = neighbor.g
                        aux.h = neighbor.h
                        aux.parent = neighbor.parent
                else:
                    #print("inserindo open_list...")
                    heapq.heappush(open_list, neighbor)

                '''if neighbor not in open_list or g_score < neighbor.g:
                    neighbor.g = g_score
                    neighbor.h = h_score
                    neighbor.parent = current_node
                    if neighbor not in open_list:
                        heapq.heappush(open_list, neighbor)'''
        
        return None