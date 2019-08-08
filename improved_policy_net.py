import numba
import numpy as np
import timeit
from numba import njit
from numba import jit
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
import os
os.chdir('/home/tony/Desktop/chess_net')
from chess_functions import *

'''
    policy funtion f(b_n+1,b_n,b_n+1) = probability b_n+1 branch will increase score over next five moves. 
'''


def buildModel2(c1,k1,c2,k2,fc,do):
    model = Sequential()
    model.add(Conv2D(c1, kernel_size=(k1,k1), activation='relu', padding='valid',input_shape=(5,8,8)))
    model.add(Conv2D(c2, kernel_size=(k2,k2), padding = 'same',activation='relu'))
    model.add(Flatten())
    model.add(Dense(fc, activation='relu'))
    model.add(Dropout(do))
    model.add(Dense(fc/2, activation='relu'))
    model.add(Dropout(do))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mae'])
    return model

model3 = buildModel2(128,2,128,2,200,.001)


def buildModel2(c1,k1,c2,k2,fc,do):
    model = Sequential()
    model.add(Conv2D(c1, kernel_size=(k1,k1), activation='relu', padding='valid',input_shape=(5,8,8)))
    model.add(Conv2D(c2, kernel_size=(k2,k2), padding = 'same',activation='relu'))
    model.add(Flatten())
    model.add(Dense(fc, activation='relu'))
    model.add(Dropout(do))
    model.add(Dense(fc/2, activation='relu'))
    model.add(Dropout(do))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='nadam', loss='mean_absolute_error', metrics=['mae'])
    return model




class node:
    def __init__(self,board,turn,player,beta1,beta2,parent = [],root = False):
        self.parent = parent
        self.turn = turn
        self.model = player
        self.score_0 = score(board,beta1,beta2)              #score is with respect to player 0 
        self.score_1 = -score(board,beta1,beta2)                  #score is with respect to player 1. 
        self.children = []
        self.projected_probability = 0
        self.num_visits = 1
        self.descendents_calculated = 1
        self.state = board
        self.terminal = False
        self.visited = False
        self.find_data_visits = 1
        self.observed_probability = 0
        self.score_parameters = (beta1,beta2)
    
        if root:
            pass

        else:
            if turn ==1:
                b = self.state
                bp = self.parent.state
            else:
                b = flipBoard(self.state)
                bp = flipBoard(self.parent.state)
            self.projected_probability = self.model.predict(np.array([[b,bp,b,bp,b]]))
            current_node = self
            
            
            if isCheckMate(b):
                self.terminal = True
                self.descendents_calculated = 10000000000 #discourages returning to this node
            elif isStaleMate(b):
                self.terminal = True
                self.descendents_calculated = 10000000000
            else:
                n = 0
                while True:
                    if current_node.parent.turn == 0:
                        current_node.observed_probability +=  self.score_0 > current_node.parent.score_0
                         
                    else:
                        current_node.observed_probability +=  self.score_1 > current_node.parent.score_1
                    current_node.projected_probability += ((-1)**(n+1))*self.projected_probability
                    current_node.descendents_calculated += 1
                    n+=1
                    current_node = current_node.parent
                    if not current_node.parent or n > 7:
                        break

            

    def find_children(self,number = 10):
        if self.turn == 1:
            B = flipBoard(self.state)
        else:
            B = self.state
        moves = legalMoves(B)
        
        if len(moves)<number:
            number = len(moves)
        
        x_vec = np.array([[makeMove(B,m),B,makeMove(B,m),B,makeMove(B,m)] for m in moves])
        pred = self.model.predict(x_vec)
        children = []
        
        for ndx in range(number):
            mve = np.argmax(pred)
            children.append(moves[mve])
            pred[mve] = -10
        return children


    def add_children(self,num_childs = 20):
        if self.terminal == True:
            pass
        else:
            if self.turn == 1:
                B = flipBoard(self.state)
            else:
                B = self.state
            new_turn = self.turn == 0
            moves = self.find_children(number = num_childs)
            
            if self.turn == 1:
                self.children = [node(flipBoard(makeMove(B,m)),new_turn,self.model,self.score_parameters[0],self.score_parameters[1],parent = self) for m in moves]
            else:
                self.children = [node(makeMove(B,m),new_turn,self.model,self.score_parameters[0],self.score_parameters[1],parent = self) for m in moves]


def u_value(nod):
    return nod.observed_probability/nod.descendents_calculated + np.sqrt(2*nod.parent.descendents_calculated/nod.descendents_calculated)

def u_value_actual(nod):
    return np.abs(nod.observed_probability/nod.descendents_calculated) + np.sqrt(2*nod.parent.find_data_visits/nod.find_data_visits)

def find_leaf(Tree):
    while True:
        if len(Tree.children) == 0:
            return Tree
        u_vals = [u_value(nod) for nod in Tree.children]
        Tree = Tree.children[np.argmax(u_vals)]



def exploration(Tree,iterations,width = 10):
    for ndx in range(iterations):
        w = width
        while iterations <12000:
            w = 10
        leaf = find_leaf(Tree)
        leaf.add_children(num_childs=w)
        if ndx%100 ==0 and ndx != 0:
            print(ndx)
    return Tree

def get_data_from_tree(tree,num_searches):
    x_vec = []
    y_vec = []

    for ndx in range(num_searches):
        current_tree = tree
        while True:
            for child in current_tree.children:
                if child.visited:

                    pass
                else:
                    if current_tree.turn == 1:
                        bp = flipBoard(tree.state)
                        b = flipBoard(child.state)
                    else:
                        bp = tree.state
                        b = child.state
                    if child.descendents_calculated < 20:

                        pass
                    else:
                        y_vec.append(child.observed_probability/child.descendents_calculated)
                        x_vec.append([b,bp,b,bp,b])
                       
                child.visited = True
            u_vals = [u_value_actual(child) for child in current_tree.children]
            current_tree = current_tree.children[np.argmax(u_vals)]
            if len(current_tree.children) == 0:
                current_tree.find_data_visits = 100000000
                break

            current_tree.find_data_visits += 1
    #x = []
    #y = []
    #for ndx in range(len(y_vec)):
    #    if y_vec[ndx] != 0:
    #        x.append(x_vec[ndx])
    #        y.append(y_vec[ndx])
    return x_vec,y_vec


def play_game_vs_random(model,searches = 50,w = 1):
    board = createBoard()
    for ndx in range(50):
        tree = node(board,0,model,0,0,root = True)
        tree = exploration(tree,searches,width = w)
        pred = [child.observed_probability/child.descendents_calculated for child in tree.children]
        board = tree.children[np.argmax(pred)].state
        print(board)
        print('')
        if isCheckMate(board):
            print('AI wins :)')
            break
        board = flipBoard(board)

        mve = randomMove(board)
        board = makeMove(board,mve)
        if isCheckMate(board) or isStaleMate(board):
            break
        board = flipBoard(board)
        print(board)
        print('')
        print(score(board,0,0))


def play_game(model,model2,search1,search2,beta1,beta2,beta11,beta22):
    board = createBoard()

    for ndx in range(50):
        tree = node(board,0,model,beta1,beta2,root = True)
        tree = exploration(tree,search1,width = 3)
        pred = [child.observed_probability/child.descendents_calculated for child in tree.children]
        board = tree.children[np.argmax(pred)].state
        print(board)
        print('')
        if isCheckMate(board) or isStaleMate(board):
            print('AI wins :)')
            break
        board = flipBoard(board)
        print(len(legalMoves(board)))
        tree = node(board,0,model2,beta11,beta22,root = True)
        tree = exploration(tree,search2,width = 3)
        pred = [child.observed_probability/child.descendents_calculated for child in tree.children]
        board = tree.children[np.argmax(pred)].state
        if isCheckMate(board) or isStaleMate(board):
            print('Ai 2 wins (or stale mate)')
            break
        board = flipBoard(board)
        print(len(legalMoves(board)))
        print(board)
        print('')
        print(score(board,beta1,beta2))
        print(score(board,beta11,beta22))
        print(score(board,0,0))


len(y_vec)


x_vec = []
y_vec= []
for ndx in range(len(y)):
    if y[ndx] != 0 or ndx%5 == 0:
        x_vec.append(x[ndx])
        y_vec.append(y[ndx])





model = buildModel2(128,2,128,2,200,.001)
model2 = buildModel2(128,2,256,2,200,.001)
model3 = buildModel2(128,2,256,2,200,.001)

board = createBoard()
tree = node(board,0,model,0,0,root = True)
explore = 80000
tree = exploration(tree,explore,width = 30)

x_vec,y_vec = get_data_from_tree(tree,100000)
model.fit(np.array(x),np.array(y),epochs = 60)

explore = 6000
board = createBoard()
x_vec = []
y_vec = []
moves = 2
for ndx in range(3):
    tree = node(board,0,model,0,0,root = True)
    tree = exploration(tree,explore,width = 10)
    pred = [child.observed_probability/child.descendents_calculated for child in tree.children]
    board = tree.children[np.argmax(pred)].state
    x,y = get_data_from_tree(tree,explore)
    x_vec.extend(x)
    y_vec.extend(y)
    board = flipBoard(board)
    tree = node(board,0,model,0,0,root = True)
    tree = exploration(tree,explore,width = 10)
    pred = [child.observed_probability/child.descendents_calculated for child in tree.children]
    board = tree.children[np.argmax(pred)].state
    x,y = get_data_from_tree(tree,explore)
    x_vec.extend(x)
    y_vec.extend(y)
    board = flipBoard(board)


1+1
len(y)