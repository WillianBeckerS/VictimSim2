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

    def __init__(self, env, config_file, resc, id):
        """ Construtor do agente random on-line
        @param env: a reference to the environment 
        @param config_file: the absolute path to the explorer's config file
        @param resc: a reference to the rescuer agent to invoke when exploration finishes
        """
        Explorer.contador_instancias += 1
        super().__init__(env, config_file)

        self.id = id

        self.control = 0
        self.path = Stack()

        self.base = Node(env.dic["BASE"][0], env.dic["BASE"][1])
        print(self.base.x, self.base.y)
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

        self.width = env.dic["GRID_WIDTH"]
        self.height = env.dic["GRID_HEIGHT"]
        print(self.width, self.height)
        
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
        '''if (self.x, self.y) not in self.untried:
            self.untried[(self.x, self.y)] = list(range(8))
            random.shuffle(self.untried[(self.x, self.y)])'''

        if (self.x, self.y) not in self.untried:
            if self.id == 1:
                self.untried[(self.x, self.y)] = [0, 7, 1, 4, 2, 3, 5, 6]
            elif self.id == 2:
                self.untried[(self.x, self.y)] = [2, 1, 3, 6, 4, 5, 0, 7]
            elif self.id == 3:
                self.untried[(self.x, self.y)] = [4, 3, 5, 0, 1, 2, 6, 7]
            elif self.id == 4:
                self.untried[(self.x, self.y)] = [6, 5, 7, 2, 3, 1, 4, 0]

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
                if (self.x + dx, self.y + dy) in self.untried:
                    self.untried[(self.x, self.y)].pop(0)
                    continue
                # Save the movement result
                self.results[(self.x, self.y)] = {(Explorer.AC_INCR[self.untried[(self.x, self.y)][0]]): (self.x + dx, self.y + dy)}
                if (self.x, self.y) not in self.unbacktracked:
                    self.unbacktracked[(self.x, self.y)] = []
                    # Save the movement in the unbacktracked dictionary
                self.unbacktracked[(self.x, self.y)].append((self.x + dx, self.y + dy))
                return Explorer.AC_INCR[self.untried[(self.x, self.y)].pop(0)]
                
        # Check if all movements was already tried
        if not self.untried[(self.x, self.y)]:
            if (self.x, self.y) in self.unbacktracked:
                if self.unbacktracked[(self.x, self.y)]:
                    return self.unbacktracked[(self.x, self.y)].pop(0)
            return random.choice([-1, 0, 1]), random.choice([-1, 0, 1])

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
            #print(f"{self.NAME} {self.id}: Wall or grid limit reached at ({self.x + dx}, {self.y + dy})")

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
                print(f"{self.NAME} {self.id} Victim found at ({self.x}, {self.y}), rtime: {self.get_rtime()}")
                #print(f"{self.NAME} {self.id} Seq: {seq} Vital signals: {vs}")
            
            # Calculates the difficulty of the visited cell
            difficulty = (rtime_bef - rtime_aft)
            if dx == 0 or dy == 0:
                difficulty = difficulty / self.COST_LINE
            else:
                difficulty = difficulty / self.COST_DIAG

            # Update the map with the new cell
            self.map.add((self.x, self.y), difficulty, seq, self.check_walls_and_lim())
            #print(f"{self.NAME} {self.id}:at ({self.x}, {self.y}), diffic: {difficulty:.2f} vict: {seq} rtime: {self.get_rtime()}")

        return

    def come_back(self):
        dx, dy = self.walk_stack.pop()
        dx = dx * -1
        dy = dy * -1

        result = self.walk(dx, dy)
        if result == VS.BUMPED:
            print(f"{self.NAME} {self.id}: when coming back bumped at ({self.x+dx}, {self.y+dy}) , rtime: {self.get_rtime()}")
            return
        
        if result == VS.EXECUTED:
            # update the agent's position relative to the origin
            self.x += dx
            self.y += dy
            #print(f"{self.NAME} {self.id}: coming back at ({self.x}, {self.y}), rtime: {self.get_rtime()}")

    def come_back_Astar(self):
        
        dx, dy = self.path.pop()

        obstacules = self.check_walls_and_lim()
        '''for key, incr in Explorer.AC_INCR.items():
            if (obstacules[key] == VS.WALL or obstacules[key] == VS.END) and incr[0] == dx and incr[1] == dy:
                self.map.add((self.x + dx, self.y + dy), VS.OBST_WALL, VS.NO_VICTIM, self.check_walls_and_lim())
                self.path.items.clear()
                self.control = 0
                print("\n\n\nRECALCULANDO A*\n\n\n")
                return'''
                
        result = self.walk(dx, dy)
        if result == VS.BUMPED:
            print(f"{self.NAME} {self.id}: when coming back bumped at ({self.x+dx}, {self.y+dy}) , rtime: {self.get_rtime()}")
            return
        
        if result == VS.EXECUTED:
            # update the agent's position relative to the origin
            self.x += dx
            self.y += dy
            #print(f"{self.NAME} {self.id}: coming back at ({self.x}, {self.y}), rtime: {self.get_rtime()}")

    def joinMaps(self, maps):
        joinedMap = Map()
        for i in Explorer.maps:
            joinedMap.map_data.update(i.map_data)

        return joinedMap

    def deliberate(self) -> bool:
        """ The agent chooses the next action. The simulator calls this
        method at each cycle. Must be implemented in every agent"""

        # consumed_time = self.TLIM - self.get_rtime()
        # if consumed_time < self.get_rtime():
        #     self.explore()
        #     return True
        
        heuristic = self.chebyshev(Node(self.x, self.y), Node(0, 0))
        if self.control == 0 and self.get_rtime() > 30*heuristic:
            self.explore()
            return True

        # time to come back to the base
        if self.walk_stack.is_empty() or (self.x == 0 and self.y == 0):
            # time to wake up the rescuer
            # pass the walls and the victims (here, they're empty)
            print(f"{self.NAME} {self.id}: rtime {self.get_rtime()}, invoking the rescuer")
            print(f"{self.NAME} {self.id}: rtime {self.get_rtime()}, instancias: " + str(Explorer.contador_instancias))
            #input(f"{self.NAME} {self.id}: type [ENTER] to proceed")
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
            self.astar((self.x, self.y), (0, 0))
            print(f"{self.NAME} {self.id} A* path: " + str(len(self.path.items)) + ' '.join(str(x) for x in self.path.items) )
            #for i in path:
                #print("pop: " + str(path.pop()))
            self.control = 1

        self.come_back_Astar()
        return True

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
        for key, incr in Explorer.AC_INCR.items():
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