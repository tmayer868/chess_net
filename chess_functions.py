import numba
import numpy as np
from numba import njit
from numba import jit

'''
small library of chess functions.
castling and enpassent still not implemneted.


'''



@jit(nopython = True)
def createBoard():
    return np.array([[-5,-3,-4,-9,-10,-4,-3,-5],[-1,-1,-1,-1,-1,-1,-1,-1],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[1,1,1,1,1,1,1,1],[5,3,4,10,9,4,3,5]])



@jit(nopython = True)
def flipBoard(B):

    newBoard = np.zeros((8,8))
    for ndx in range(8):
        newBoard[ndx] = B[7-ndx]
    newBoard = -newBoard
    newerBoard = np.zeros((8,8),dtype = np.int64)
    for ndx in range(8):
        newerBoard[:,ndx] = newBoard[:,7-ndx]

    return newerBoard





@jit(nopython = True)
def obstructedPath(B,x0,y0,x1,y1):
    '''
        Determines if move is obstructed

    '''
    if B[x0,y0] == 3: # 3 represents knight so no need to worry about path
        return False

    if x1 == x0:
        pLength = int(np.abs(y1 - y0) - 1)
        direction = int(np.sign(y1-y0))
        for ndx in range(pLength):
            if B[x0,y0 + direction*(ndx+1)] != 0:
                return True
    if y1 == y0:
        pLength = int(np.abs(x1 - x0) - 1)
        direction = int(np.sign(x1-x0))
        for ndx in range(pLength):
            if B[x0 + direction*(ndx+1),y0] != 0:
                return True
    else:
        deltax = int(np.abs(x1-x0) - 1)
        xDir = int(np.sign(x1-x0))
        yDir = int(np.sign(y1-y0))
        for ndx in range(deltax):
            if B[x0 + xDir*(ndx+1),y0 + yDir*(ndx+1)] != 0:
                return True
    return False



@jit(nopython = True)
def psudoLegal(B,x0,y0,x1,y1):


    if B[x0,y0] <= 0:
        return False
    if B[x1,y1] > 0:
        return False
    if obstructedPath(B,x0,y0,x1,y1):
        return False

    piece = B[x0,y0]


    if piece == 1:
        if x0 - x1 > 0 and x0 - x1 <= 2 and B[x1,y1] == 0 and y1 == y0:
            return True
        if x0 - x1 == 1 and int(np.abs(y0 - y1)) == 1 and B[x1,y1] <0:
            return True
        else:
            return False
    if piece == 2: # 2 represents a pawn that has already moved at least once
        if x0 - x1 == 1 and B[x1,y1] == 0 and y1 == y0:
            return True
        if x0 - x1 == 1 and int(np.abs(y1 - y0)) == 1 and B[x1,y1] < 0:
            return True
        else:
            return False

    if piece == 5: # rook
        if x0 != x1 and y0 != y1:
            return False
        else:
            return True
    if piece == 3: # knight
        if int(np.abs(x1 - x0)) + int(np.abs(y1 - y0)) != 3 or (int(np.abs(x1 - x0)) != 1 and int(np.abs(y1 - y0)) != 1):
            return False
        else:
            return True
    if piece == 4: # Bishop
        if int(np.abs(x0-x1)) != int(np.abs(y0 - y1)):
            return False
        else:
            return True
    if piece == 9: #Queen
        if x0 == x1 or y0 == y1 or int(np.abs(x0-x1)) == int(np.abs(y0 - y1)):
            return True
        else:
            return False

    if piece == 10: #King
        if int(np.abs(x0 - x1)) < 2 and int(np.abs(y0-y1)) < 2:
            return True
        else:
            return False




@jit(nopython = True)
def findKing(B):
    for ndx1 in range(8):
        for ndx2 in range(8):
            if B[ndx1,ndx2] == -10:
                return ndx1,ndx2



@jit(nopython = True)
def legalMove(Board,x0,y0,x1,y1):
    if x0 == x1 and y0 == y1:
        return False

    B = np.copy(Board)
    piece = B[x0,y0]
    if psudoLegal(B,x0,y0,x1,y1) == False:
        return False
    B[x1,y1] = piece
    B[x0,y0] = 0
    newBoard = flipBoard(B)
    kingX,kingY = findKing(newBoard)

    for ndx1 in range(8):
        for ndx2 in range(8):
            if psudoLegal(newBoard,ndx1,ndx2,kingX,kingY):
                return False
    if B[x1,y1] == -10:
        print('something wrong check legalMove')
        return False
    return True


@jit(nopython = True)
def makeMove(Board,m):
    B = np.copy(Board)
    x0,y0,x1,y1 = m
    if B[x0,y0] == 1:
        B[x0,y0] = 2
    B[x1,y1] = B[x0,y0]
    B[x0,y0] = 0
    if x1 == 0 and B[x1,y1] == 2:
        B[x1,y1] = 9
    return B



@jit(nopython = True)
def isCheck(Board):
    #Checks if opposing player is in check
    kingX , kingY = findKing(Board)
    for indx1 in range(8):
        for indx2 in range(8):
            if psudoLegal(Board,indx1,indx2,kingX,kingY):
                return True
    return False



@jit(nopython = True)
def isCheckMate(B):
    #checks to see if player who just moved, before board flip, won
    if isCheck(B) == True and len(legalMoves(flipBoard(B))) == 0:
        return True
    else:
        return False



@jit(nopython = True)
def legalMoves(B):
    moves = []
    for ndx1 in range(8):
        for ndx2 in range(8):
            for ndx3 in range(8):
                for ndx4 in range(8):
                    if legalMove(B,ndx1,ndx2,ndx3,ndx4):
                        moves.append((ndx1,ndx2,ndx3,ndx4))
    return moves



@jit(nopython = True)
def isStaleMate(B):
    if isCheck(B) == False and len(legalMoves(flipBoard(B))) == 0:
        return True
    else:
        return False





@jit(nopython = True)
def randomMove(B):
    moves = legalMoves(B)
    n = np.random.randint(len(moves))
    return moves[n]



@njit
def checkForLoop(gameBoards,moves):
    if np.all(gameBoards[moves -1 ] == gameBoards[moves - 5 ]) and np.all(gameBoards[moves - 9] == gameBoards[moves - 5 ]):
        return True
    else:
        return False


'''
@njit
def score(B):
    scoreTurn = 0
    scoreOppos = 0
    for ndx1 in range(8):
        for ndx2 in range(8):
            piece = B[ndx1][ndx2]
            if piece == 1:
                scoreTurn +=1
            if piece == 2:
                scoreTurn +=1
            if piece == 5:
                scoreTurn += 5
            if piece == 9:
                scoreTurn += 9
            if piece == 4:
                scoreTurn += 3
            if piece == 3:
                scoreTurn += 3
            if piece == -1:
                scoreOppos +=1
            if piece == -2:
                scoreOppos +=1
            if piece == -5:
                scoreOppos += 5
            if piece == -9:
                scoreOppos += 9
            if piece == -4:
                scoreOppos += 3
            if piece == -3:
                scoreOppos += 3
    return (scoreTurn - scoreOppos)/(scoreTurn + scoreOppos)
'''

@njit
def score(b,beta1,beta2):
    '''
        a more sophisticated measure of a board postion. 

    '''
    white_legal_moves = 0
    black_legal_moves = 0
    white_controlled_squares = 0
    black_controlled_squares = 0
    white_material = 0
    black_material = 0
    for ndx1 in range(8):
        for ndx2 in range(8):
            piece = b[ndx1][ndx2]
            if piece == 1:
                white_material +=1
            if piece == 2:
                white_material +=1
            if piece == 5:
                white_material += 5
            if piece == 9:
               white_material += 9
            if piece == 4:
                white_material += 3
            if piece == 3:
                white_material += 3
            if piece == -1:
                black_material +=1
            if piece == -2:
                black_material +=1
            if piece == -5:
                black_material += 5
            if piece == -9:
                black_material += 9
            if piece == -4:
                black_material += 3
            if piece == -3:
                black_material += 3

    white_legal_moves = len(legalMoves(b))
    black_legal_moves = len(legalMoves(flipBoard(b)))
    for ndx1 in range(8):
        for ndx2 in range(8):
            for ndx3 in range(8):
                for ndx4 in range(8):
                    if psudoLegal(b,ndx1,ndx2,ndx3,ndx4):
                        white_controlled_squares += 1
                    if psudoLegal(flipBoard(b),ndx1,ndx2,ndx3,ndx4):
                        black_controlled_squares += 1
    numerator = (white_material - black_material + beta1*( white_controlled_squares - black_controlled_squares) + beta2*(white_legal_moves - black_legal_moves))

    denominator = (white_material + black_material + beta1*( white_controlled_squares + black_controlled_squares) + beta2*(white_legal_moves + black_legal_moves))
    
    return numerator/denominator