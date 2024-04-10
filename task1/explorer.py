# EXPLORER AGENT
# @Author: Tacla, UTFPR
#
### It walks randomly in the environment looking for victims. When half of the
### exploration has gone, the explorer goes back to the base.

import sys
import os
import random
import math
from abc import ABC, abstractmethod
from vs.abstract_agent import AbstAgent
from vs.constants import VS
from map import Map
import heapq

class Stack:
    def __init__(self):
        self.items = []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        if not self.is_empty():
            return self.items.pop()

    def is_empty(self):
        return len(self.items) == 0

class Node:
    def __init__(self, x, y, parent=None):
        self.x = x
        self.y = y
        self.parent = parent
        self.g = 0
        self.h = 0
    
    def __lt__(self, other):
        return (self.g + self.h) < (other.g + other.h)

class Explorer(AbstAgent):
    contador_instancias = 0
    victimsTotals = []
    maps = []


    def __init__(self, env, config_file, resc):
        """ Construtor do agente random on-line
        @param env: a reference to the environment 
        @param config_file: the absolute path to the explorer's config file
        @param resc: a reference to the rescuer agent to invoke when exploration finishes
        """
        Explorer.contador_instancias += 1
        super().__init__(env, config_file)

        self.open_list = []
        self.closed_set = set()
        self.base_node = Node(0, 0)

        self.control = 0
        self.desempilhando = 0
        self.flagPop = 0
        self.current_node = Node(0,0)

        self.walk_stack = Stack()  # a stack to store the movements
        self.set_state(VS.ACTIVE)  # explorer is active since the begin
        self.resc = resc           # reference to the rescuer agent
        self.x = 0                 # current x position relative to the origin 0
        self.y = 0                 # current y position relative to the origin 0
        self.map = Map()           # create a map for representing the environment
        self.victims = {}          # a dictionary of found victims: (seq): ((x,y), [<vs>])
                                   # the key is the seq number of the victim,(x,y) the position, <vs> the list of vital signals

        self.results = {}   # a dictionary of movement results: (i, j): (AC_INCR[n]: (l, c)), where 0 <= n <= 7
        self.untried = {} # a dictionary for untried movements
        self.unbacktracked = {} # a dictionary for saving unbacktracking
        
        # Put the current position - the base - in the map
        self.map.add((self.x, self.y), 1, VS.NO_VICTIM, self.check_walls_and_lim())

    def __del__(self):
        Explorer.contador_instancias -= 1

    def get_next_position(self):
        """ Randomically, gets the next position that can be explored (no wall and inside the grid)
            There must be at least one CLEAR position in the neighborhood, otherwise it loops forever.
        """
        # Check the neighborhood walls and grid limits
        obstacles = self.check_walls_and_lim()
    
        # Loop until a CLEAR position is found
        while True:
            # Get a random direction
            direction = random.randint(0, 7)
            # Check if the corresponding position in walls_and_lim is CLEAR
            if obstacles[direction] == VS.CLEAR:
                return Explorer.AC_INCR[direction]
        
    
    def online_dfs(self):
        """ Implements Online DFS to explore the environment
        """

        # Check the neighborhood walls and grid limits
        obstacles = self.check_walls_and_lim()

        # A randomic sequence for moving
        if (self.x, self.y) not in self.untried:
            self.untried[(self.x, self.y)] = list(range(8))
            random.shuffle(self.untried[(self.x, self.y)])

        # Loop while untried list is not empty
        while self.untried[(self.x, self.y)]:
            # Next movement
            dx, dy = Explorer.AC_INCR[self.untried[(self.x, self.y)][0]]
            # If colide with a wall or reached the grid limit
            if obstacles[self.untried[(self.x, self.y)][0]] == VS.WALL or obstacles[self.untried[(self.x, self.y)][0]] == VS.END:
                # Save the movement result
                self.results[(self.x, self.y)] = {(Explorer.AC_INCR[self.untried[(self.x, self.y)][0]]): (self.x, self.y)}
                # Remove from untried list
                self.untried[(self.x, self.y)].pop(0)
            # Check if the corresponding position in walls_and_lim is CLEAR
            elif obstacles[self.untried[(self.x, self.y)][0]] == VS.CLEAR:
                # Save the movement result
                self.results[(self.x, self.y)] = {(Explorer.AC_INCR[self.untried[(self.x, self.y)][0]]): (self.x + dx, self.y + dy)}
                if (self.x, self.y) not in self.unbacktracked:
                    self.unbacktracked[(self.x, self.y)] = []
                    # Save the movement in the unbacktracked dictionary
                self.unbacktracked[(self.x, self.y)].append((self.x + dx, self.y + dy))
                return Explorer.AC_INCR[self.untried[(self.x, self.y)].pop(0)]
                
        # Check if all movements was already tried
        if not self.untried[(self.x, self.y)]:
            return self.unbacktracked[(self.x, self.y)].pop(0)

    def explore(self):
        # get an random increment for x and y       
        dx, dy = self.online_dfs()

        # Moves the body to another position
        rtime_bef = self.get_rtime()
        result = self.walk(dx, dy)
        rtime_aft = self.get_rtime()

        # Test the result of the walk action
        # Should never bump, but for safe functionning let's test
        if result == VS.BUMPED:
            # update the map with the wall
            self.map.add((self.x + dx, self.y + dy), VS.OBST_WALL, VS.NO_VICTIM, self.check_walls_and_lim())
            #print(f"{self.NAME}: Wall or grid limit reached at ({self.x + dx}, {self.y + dy})")

        if result == VS.EXECUTED:
            # check for victim returns -1 if there is no victim or the sequential
            # the sequential number of a found victim
            self.walk_stack.push((dx, dy))

            # update the agent's position relative to the origin
            self.x += dx
            self.y += dy          

            # Check for victims
            seq = self.check_for_victim()
            if seq != VS.NO_VICTIM:
                vs = self.read_vital_signals()
                self.victims[vs[0]] = ((self.x, self.y), vs)
                print(f"{self.NAME} Victim found at ({self.x}, {self.y}), rtime: {self.get_rtime()}")
                #print(f"{self.NAME} Seq: {seq} Vital signals: {vs}")
            
            # Calculates the difficulty of the visited cell
            difficulty = (rtime_bef - rtime_aft)
            if dx == 0 or dy == 0:
                difficulty = difficulty / self.COST_LINE
            else:
                difficulty = difficulty / self.COST_DIAG

            # Update the map with the new cell
            self.map.add((self.x, self.y), difficulty, seq, self.check_walls_and_lim())
            #print(f"{self.NAME}:at ({self.x}, {self.y}), diffic: {difficulty:.2f} vict: {seq} rtime: {self.get_rtime()}")

        return

    def come_back(self):
        dx, dy = self.walk_stack.pop()
        dx = dx * -1
        dy = dy * -1

        result = self.walk(dx, dy)
        if result == VS.BUMPED:
            print(f"{self.NAME}: when coming back bumped at ({self.x+dx}, {self.y+dy}) , rtime: {self.get_rtime()}")
            return
        
        if result == VS.EXECUTED:
            # update the agent's position relative to the origin
            self.x += dx
            self.y += dy
            #print(f"{self.NAME}: coming back at ({self.x}, {self.y}), rtime: {self.get_rtime()}")

    def joinMaps(self, maps):
        joinedMap = Map()
        for i in Explorer.maps:
            joinedMap.map_data.update(i.map_data)

        return joinedMap

    def deliberate(self) -> bool:
        """ The agent chooses the next action. The simulator calls this
        method at each cycle. Must be implemented in every agent"""

        consumed_time = self.TLIM - self.get_rtime()
        if consumed_time < self.get_rtime():
            self.explore()
            return True

        # time to come back to the base
        if self.walk_stack.is_empty() or (self.x == 0 and self.y == 0):
            # time to wake up the rescuer
            # pass the walls and the victims (here, they're empty)
            print(f"{self.NAME}: rtime {self.get_rtime()}, invoking the rescuer")
            print(f"{self.NAME}: rtime {self.get_rtime()}, instancias: " + str(Explorer.contador_instancias))
            #input(f"{self.NAME}: type [ENTER] to proceed")
            Explorer.maps.append(self.map)
            Explorer.victimsTotals.append(self.victims)
            Explorer.contador_instancias -= 1
            if(Explorer.contador_instancias == 0):
                victims = {}
                for i in Explorer.victimsTotals:
                    victims.update(i)
                #self.resc.make_groups_victims(victims)
                combinedMap = self.joinMaps(Explorer.maps)
                self.resc.receive_map_victims(combinedMap, victims)
            
            return False

        if(self.control == 0):
            #print("A* path: " + ' '.join(str(x) for x in self.astar((self.x, self.y), (0, 0))) )
            start_node = Node(self.x, self.y)
            print("celula inicial: " + str(start_node.x) + " " + str(start_node.y))
            heapq.heappush(self.open_list, start_node)
            self.control = 1

        self.astar()
        #self.come_back()
        return True

    def chebyshev(self, node, end_node):      # heuristica
        return max(abs(node.x - end_node.x), abs(node.y - end_node.y))

    def get_neighbors(self, node):
        neighbors = []
        #directions = [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]
        obstacules = self.check_walls_and_lim()        
        print(' '.join(str(x) for x in obstacules))
        #i = 0
        for key, incr in Explorer.AC_INCR.items():
            new_x, new_y = node.x + incr[0], node.y + incr[1]

            print("new node: " + str(new_x) + " " + str(new_y))
            if obstacules[key] != VS.WALL and obstacules[key] != VS.END:
                print("obstacules aceitos: " + str(obstacules[key]))
                neighbors.append(Node(new_x, new_y, node))
        return neighbors

    def adjc(self, c1, c2):
        x1, y1 = c1
        x2, y2 = c2

        # Verifica se as coordenadas estão adjacentes em termos de distância Manhattan
        if abs(x1 - x2) <= 1 and abs(y1 - y2) <= 1:
            return True
        else:
            return False

    def astar(self):
        if len(self.open_list) != 0:
            for i in self.open_list:
                print(str(i.x) + " x " + str(i.y) + " - ")

            if self.desempilhando == 0:
                if self.flagPop != 1:
                    self.current_node = heapq.heappop(self.open_list)
                else:    
                    self.flagPop = 0

                print("current node: " + str(self.current_node.x) + " x " + str(self.current_node.y))
                if(self.adjc((self.x, self.y), (self.current_node.x, self.current_node.y))):
                    dx = self.current_node.x - self.x
                    dy = self.current_node.y - self.y
                    print(str(dx) + " x " + str(dy))
                    print("no atual: " + str(self.x) + " " + str(self.y))
                    self.walk(dx, dy)
                    # update the position
                    self.x += dx
                    self.y += dy 
                    self.walk_stack.push((dx, dy))

                    if self.current_node.x == self.base_node.x and self.current_node.y == self.base_node.y:
                        return  # chegou na base
                    
                    self.closed_set.add((self.current_node.x, self.current_node.y))
                    
                    for neighbor in self.get_neighbors(self.current_node):
                        if neighbor in self.closed_set:
                            continue
                        
                        g_score = self.current_node.g + 1
                        h_score = self.chebyshev(neighbor, self.base_node)
                        #f_score = g_score + h_score
                        
                        #print("scores: " + str(g_score) + " " + str(h_score) + " " + str(f_score) )
                        neighbor.g = g_score
                        neighbor.h = h_score
                        neighbor.parent = self.current_node

                        aux = next((obj for obj in self.open_list if obj.x == neighbor.x and obj.y == neighbor.y), None)

                        if aux is not None:
                            if neighbor.g < aux.g:
                                aux.g = neighbor.g
                                aux.h = neighbor.h
                                aux.parent = neighbor.parent
                        else:
                            print("inserindo open_list...")
                            heapq.heappush(self.open_list, neighbor)
                else:
                    self.desempilhando = 1
            else:
                print("desempilhando...")

                dx, dy = self.walk_stack.pop()
                dx = dx * -1
                dy = dy * -1

                self.walk(dx, dy)

                # update the agent's position relative to the origin
                self.x += dx
                self.y += dy 

                if(self.adjc((self.x, self.y), (self.current_node.x, self.current_node.y))):
                    self.desempilhando = 0
                    self.flagPop = 1
        else:
            print("There is no path...")
        
        return None