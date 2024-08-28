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
import numpy as np
from tflite_runtime.interpreter import Interpreter
from sklearn.preprocessing import StandardScaler
import joblib

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
    population_size = 100
    generations = 500
    mutation_rate = 0.01

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

        self.dir_clusters = './clusters'
        Rescuer.rescuers.append(self)
        # Starts in IDLE state.
        # It changes to ACTIVE when the map arrives
        self.set_state(VS.IDLE)
        self.path = Stack()
        self.load_neural_model()

    def load_neural_model(self):    # pip install tflite-runtime
        # 1. Carregar o modelo TFLite
        self.interpreter = Interpreter(model_path='model2.tflite')
        self.interpreter.allocate_tensors()

        # 2. Obter detalhes dos tensores de entrada e saída
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # 3. Carregar o scaler usado durante o treinamento
        self.scaler = joblib.load('scaler_vitimas.pkl')

    def prediction(self, qPB, pulse, respRate):
        # 4. Definir os valores da nova entrada
        # Exemplo: qualidade da pressão arterial=2, pulso=75, frequência respiratória=20
        new_data = np.array([[qPB,pulse,respRate]])

        # 5. Normalizar a entrada usando o scaler carregado
        new_data_normalized = self.scaler.transform(new_data)

        # 6. Fazer a previsão
        self.interpreter.set_tensor(self.input_details[0]['index'], new_data_normalized.astype(np.float32))
        self.interpreter.invoke()

        # 7. Obter a previsão
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        predicted_class = np.argmax(output_data, axis=1)[0] + 1

        if predicted_class == 1:
            fitness_value = 4
        elif predicted_class == 2:
            fitness_value = 3
        elif predicted_class == 3:
            fitness_value = 2
        else:
            fitness_value = 1

        return fitness_value

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

        population = self.__initialize_population()

        print(f'POPULAAA{population}')

        for _ in range(Rescuer.generations):
            fitness_scores = [self.__evaluate_fitness(individual) for individual in population]
            selected_individuals = self.__selection(population, fitness_scores)
            next_population = []

            while len(next_population) < Rescuer.population_size:
                parent1 = random.choice(selected_individuals)
                parent2 = random.choice(selected_individuals)
                child1, child2 = self.__crossover(parent1, parent2)
                next_population.append(self.__mutate(child1, Rescuer.mutation_rate))
                next_population.append(self.__mutate(child2, Rescuer.mutation_rate))

            population = next_population

        best_individual = max(population, key=self.__evaluate_fitness)

        print(f'best plan: {best_individual}')
        
        # Calcular A* de best_individual para ver se é possível completar a rota no tempo limite
        # Caso não seja possível, remover este indivíduo de population e repitir best_individual = max(population, key=self.__evaluate_fitness)

        # Ordenando self.victims de acordo com a sequencia devolvida pelo AG
        self.victims = {key: self.victims[key] for key in best_individual if key in self.victims}
        print(f'VICCCC{self.victims.items()}')
        first_victim = next(iter(self.victims.items()))
        self.astar((0, 0), (first_victim[1][0]))

        while self.path.is_empty() == False:
            dx, dy, vict = self.path.pop()
            self.plan.append((dx, dy, vict)) 

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
                dx, dy, vict = self.path.pop()
                self.plan.append((dx, dy, vict)) 
            print(f"Atual: {atual}, Próximo: {proximo}")
            
        ultima_chave, ultimo_valor = list(self.victims.items())[-1]
        #ultimo_elemento = next(iterador_proximo, None)
        #if ultimo_elemento:
        #    print(f"Último elemento: {ultimo_elemento}")

        self.astar((ultimo_valor[0][0], ultimo_valor[0][1]), (0, 0))
        while self.path.is_empty() == False:
            dx, dy, vict = self.path.pop()
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

    def __initialize_population(self):
        population = []
        individual = []

        # Population initialize based on the rescuer cluster
        print(self.clusters[0].victims.items())
        for seq, ((x, y), vs) in self.clusters[0].victims.items():
            individual.append(seq)

        print(f'individual rescuer: {individual}')

        # Gerar todas as permutações de tamanho 1 até len(individual)
        all_permutations = []
        max_permutations = 5
        for r in range(1, len(individual) + 1):
            # Misturar as sequências para evitar padrões
            random.shuffle(individual)
            # Limitar o número de permutações geradas para cada r
            limited_permutations = itertools.islice(itertools.permutations(individual, r), max_permutations)
            all_permutations.extend(list(limited_permutations))

        print(f'all_permutations: {all_permutations}')

        if len(all_permutations) < Rescuer.population_size:
            population = [list(p) for p in random.sample(all_permutations, len(all_permutations))]
        else:
            # Selecionar aleatoriamente Rescuer.population_size permutações para criar a população
            population = [list(p) for p in random.sample(all_permutations, Rescuer.population_size)]

        print(f'POPULATION (rescuer): {population}')

        return population
    
    def __evaluate_fitness(self, plan):
        """
        Evaluates the fitness of a plan based on the time spent and the severity of rescued victims.
        @param plan: list of planned actions
        @return fitness: fitness value of the plan
        """

        total_severity = 0
        total_distance = 0
        current_x, current_y = 0, 0

        for elem in plan:
            for seq, ((x, y), vs) in self.clusters[0].victims.items():
                target_x, target_y = x, y
                if elem == seq:
                    # Pegar os sinais vitais de cada vítima e passar pra rede neural
                    total_severity += self.prediction(vs[2], vs[3], vs[4]) # CONFERIR
                    #total_severity += random.randint(1, 4)
                    # Soma distancia euclidiana total de passar por todas as vitimas
                    total_distance += self.__euclidean_distance(current_x, current_y, target_x, target_y)
                    current_x, current_y = target_x, target_y
                    break
                
        # Somar distancia para voltar até a base
        total_distance += self.__euclidean_distance(current_x, current_y, 0, 0)

        # Calculate the fitness value considering the euclidean distance and the severity of rescued victims
        if total_distance != 0:
            fitness = total_severity / total_distance
        else:
            fitness = total_severity
        
        return fitness
    
    def __euclidean_distance(self, x1, y1, x2, y2):
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    def __selection(self, population, fitness_scores):
        # Ensure fitness scores are not all zero
        total_fitness = sum(fitness_scores)
        
        if total_fitness == 0:
            # If all fitness scores are zero, assign equal probability to each individual
            fitness_scores = [1] * len(population)

        selected = random.choices(population, weights=fitness_scores, k=len(population) // 2)
        return selected

    def __crossover(self, parent1, parent2):
        if len(parent1) > 1 and len(parent2) > 1:
            # Determine o tamanho mínimo e máximo entre os pais
            min_size = min(len(parent1), len(parent2))
            max_size = max(len(parent1), len(parent2))
            
            # Gerar dois pontos de corte aleatórios baseados no menor tamanho
            cut1 = random.randint(0, min_size - 1)
            cut2 = random.randint(cut1 + 1, min_size)
            
            # Criar filhos com o tamanho do pai correspondente
            child1 = [-1] * len(parent1)
            child2 = [-1] * len(parent2)
            
            # Copiar segmento dos pais para os filhos
            child1[cut1:cut2] = parent1[cut1:cut2]
            child2[cut1:cut2] = parent2[cut1:cut2]
            
            # Função para preencher os filhos com os genes restantes
            def fill_child(child, parent):
                current_pos = cut2
                for gene in parent:
                    if gene not in child:
                        while current_pos < len(child) and child[current_pos] != -1:
                            current_pos += 1
                        if current_pos >= len(child):
                            current_pos = 0
                        while current_pos < len(child) - 1 and child[current_pos] != -1:
                            current_pos += 1
                        child[current_pos] = gene
                        current_pos += 1  # Incrementa current_pos após adicionar o gene

            fill_child(child1, parent2)
            fill_child(child2, parent1)
            
            return child1, child2
        
        return parent1, parent2
    
    def __mutate(self, individual, mutation_rate):
        '''if random.random() < mutation_rate:
            mutation_index1 = random.randint(0, len(individual) - 1)
            mutation_index2 = mutation_index1
            while mutation_index2 == mutation_index1:
                mutation_index2 = random.randint(0, len(individual) - 1)
            aux = individual[mutation_index1]
            individual[mutation_index1] = individual[mutation_index2]
            individual[mutation_index2] = aux'''
        return individual

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
            with open(os.path.join(self.dir_clusters, 'cluster' + str(i) + '.txt'), 'w') as arquivo:
                for seq, ((x, y), vs) in elem.victims.items():
                    arquivo.write(str(seq) + "," + str(x) + "," + str(y) + "," + str(vs[1]) + "," + str(vs[2]) + "," + str(vs[3]) + "," + str(vs[4]) + "," + str(vs[5]) + "\n")
                            
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
                #primeiro fora do laco pois tratasse do incremento que levara a uma vitima
                if(current_node.parent is not None):
                    self.path.push((current_node.x - current_node.parent.x, current_node.y - current_node.parent.y, True))
                current_node = current_node.parent

                while current_node:
                    if(current_node.parent is not None):
                        self.path.push((current_node.x - current_node.parent.x, current_node.y - current_node.parent.y, False))
                    else:
                        self.path.push((0, 0, False))
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