import pygame
from sys import exit
import numpy as np
import copy as cp
import math
import random as rd
import time

class GameState:
    def __init__(self, board):
        self.board = board
        self.end=-1
        self.children = []
        self.parent = None
        self.parentPlay = None # (play, movtype)
        self.parentCell = None
        self.numsimulation=0
        self.mctsv=0
        self.mctss=0
        self.uct=0

    def createChildren(self, player_id):
        differentPlayBoards = []
        for i in range(len(self.board)):
            for j in range(len(self.board)):
                if self.board[j][i] == player_id:
                    moves = get_moves(self, (i,j))
                    for mov in moves:
                        if moves[mov][0]:
                            newboard = cp.deepcopy(self.board)
                            play = (i+mov[0], j+mov[1])
                            if moves[mov][1] == 1: #movtype
                                newboard[play[1]][play[0]] = player_id
                            elif moves[mov][1] == 2:
                                newboard[play[1]][play[0]] = player_id
                                newboard[j][i] = 0
                            if newboard not in differentPlayBoards:
                                differentPlayBoards.append(newboard)
                                newboard = get_and_apply_adjacent(play, newboard, player_id)
                                newState = GameState(newboard)
                                newState.parentCell = (i,j)
                                newState.parentPlay = (play, moves[mov][1])
                                newState.parent = self
                                self.children.append(newState)

def evaluatePlay_mcts(game,board,play,cell,player):
    s1=1
    s2=0.4
    s3=0.7
    s4=0.4
    soma=0
    vec=[(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
    if play[1] == 1:
        soma+=s3
        for mov in vec:
            if not (play[0][0]<1 or play[0][0]>len(game.board)-1-1 or play[0][1]<1 or play[0][1]>len(game.board)-1-1):  
                if board[play[0][0]+mov[0]][play[0][1]+mov[1]]==3-player:
                    soma+=s1
                if board[play[0][0]+mov[0]][play[0][1]+mov[1]]==player:
                    soma+=s2
    elif play[1] == 2:
        for mov in vec:
            if not (play[0][0]<1 or play[0][0]>len(game.board)-1-1 or play[0][1]<1 or play[0][1]>len(game.board)-1-1):
                if board[play[0][1]+mov[1]][play[0][0]+mov[0]]==3-player:
                    soma+=s1
                if board[play[0][1]+mov[1]][play[0][0]+mov[0]]==player:
                    soma+=s2
            if not (cell[0]<1 or cell[0]>len(game.board)-1-1 or cell[1]<1 or cell[1]>len(game.board)-1-1):  
                if board[cell[1]+mov[1]][cell[0]+mov[0]]==player:
                    soma-=s4
    return soma

def final_move(game,board,play,player):     #### função que checka se o estado não tem children
    #print(player,'final')
    gamenp=np.array(board)
    #print(gamenp,'nparray')
    if np.count_nonzero(gamenp==(3-player))==0:
        return (True,player)
    if np.count_nonzero(gamenp==(player))==0:
        return (True,3-player)
    if np.count_nonzero(gamenp==0) != 0:
                return (False,-1)
    if np.count_nonzero(gamenp==0) == 0:  
        count_p=np.count_nonzero(gamenp==player)
        count_o=np.count_nonzero(gamenp==(3-player))
        if count_p > count_o:
            return (True,player)
        if count_o > count_p:
            return (True,3-player)
    return (True,0)

def evaluatePlay_minmax(game,board,play,cell,player,values):  #### heurística para o minimax
    s1=values[0]
    s2=values[1]
    s3=values[2]
    s4=values[3]
    soma=0
    vec=[(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
    #print(player,'pre final')
    final=final_move(game,board,play,player)
    #print(final)
    if final[0]:                                        #### ver que tipo de fim é e retornar o valor 
        if final[1]==player:
            return (math.inf)
        if final[1]==3-player:
            return (-math.inf)
        if final[1]==0:
            return 0
    if play[1] == 1:
        soma+=s3
        for mov in vec:
            if not (play[0][0]<1 or play[0][0]>len(game.board)-1-1 or play[0][1]<1 or play[0][1]>len(game.board)-1-1):  
                if board[play[0][0]+mov[0]][play[0][1]+mov[1]]==3-player:
                    soma+=s1
                if board[play[0][0]+mov[0]][play[0][1]+mov[1]]==player:
                    soma+=s2
    elif play[1] == 2:
        for mov in vec:
            if not (play[0][0]<1 or play[0][0]>len(game.board)-1-1 or play[0][1]<1 or play[0][1]>len(game.board)-1-1):
                if board[play[0][1]+mov[1]][play[0][0]+mov[0]]==3-player:
                    soma+=s1
                if board[play[0][1]+mov[1]][play[0][0]+mov[0]]==player:
                    soma+=s2
            if not (cell[0]<1 or cell[0]>len(game.board)-1-1 or cell[1]<1 or cell[1]>len(game.board)-1-1):  
                if board[cell[1]+mov[1]][cell[0]+mov[0]]==player:
                    soma-=s4
    return soma


def randomplay(game, player):
    game.createChildren(player)
    r = rd.randint(0,len(game.children)-1)
    return game.children[r]

def implementar_montecarlos(game,player):
    C=1.5
    game.createChildren(player)
    bestchildren=[]
    Sbasic=100
    for child in game.children:
        childnp=np.array(child.board)
        child.mctss=Sbasic*(1+0.1*(np.count_nonzero(childnp==player)))
        child.mctsv=montecarlots(int(child.mctss),child,player)
        child.numsimulation+=child.mctss
        if len(bestchildren)<=5:
            bestchildren.append(child)
        else:
            for bchild in bestchildren:
                if child.mctsv>bchild.mctsv:
                    bestchildren.remove(bchild)
                    bestchildren.append(child)
    for child in bestchildren:
        child.mctsv+=montecarlots(int(child.mctss),child,player)
        child.numsimulation+=child.mctss
    bestchildren=[]
    for child in game.children:
        child.uct=child.mctsv+C*(math.sqrt(1/(child.numsimulation)))
        if len(bestchildren)<=3:
            bestchildren.append(child)
        else:
            for bchild in bestchildren:
                if child.uct>bchild.uct:
                    bestchildren.remove(bchild)
                    bestchildren.append(child)
    for child in bestchildren:
        child.mctsv+=montecarlots(int((child.mctss)/2),child,player)
    bestchildren=[]
    for child in game.children:
        if len(bestchildren)==0:
            bestchildren.append(child)
        else:
            if child.mctsv>bestchildren[0].mctsv:
                bestchildren.pop(0)
                bestchildren.append(child)
    return bestchildren[0]

def montecarlots(numSimulations, game,player):
    gamenp=np.array(game.board)
    E=np.count_nonzero(gamenp==player)-np.count_nonzero(gamenp==(3-player))
    for i in range(numSimulations):
        player_hid=player
        board=cp.deepcopy(game.board)
        while game.end == -1 and game.end<10:
            if player_hid == 1:
                game.createChildren(player_hid)
                board = randomplay(game,player_hid)  #para testar os algoritmos da AI é só trocar aqui a função pelo algoritmo desejado
                game = GameState(board)
                player_hid = switchPlayer(player_hid)
            elif player_hid == 2:
                game.createChildren(player_hid)
                board = randomplay(game,player_hid)  #igual ao comentario acima
                game = GameState(board)
                player_hid = switchPlayer(player_hid)
            game.end = objective_testmcts(game, player_hid)
        if game.end-10==player_hid:
            E+=500
        elif game.end-10==3-player_hid:
            E-=500
        elif game.end==player_hid:
            E+=50
        elif game.end==3-player_hid:
            E-=50
    return E

def greedy(game,player):
    bestPlay = ([], -math.inf) #random tuple where the 2nd element holds the best play evaluation and the 1st holds its board
    game.createChildren(player)
    for state in game.children: #play[0] -> (i,j) || play][1] -> movType
        board = cp.deepcopy(state.board)
        value=evaluatePlay_mcts(state,board,state.parentPlay,state.parentCell,player)
        if value > bestPlay[1]:
            if state.parentPlay[1] == 1: 
                board[state.parentPlay[0][1]][state.parentPlay[0][0]] = player
            elif state.parentPlay[1] == 2:  
                board[state.parentCell[1]][state.parentCell[0]] = 0
                board[state.parentPlay[0][1]][state.parentPlay[0][0]] = player
            board = get_and_apply_adjacent((state.parentPlay[0][0], state.parentPlay[0][1]), board, player)
            bestPlay = (board, value)
    return GameState(bestPlay[0])

def implement_minimax(game,player,playerAI):                   
    max_depth = 5 
    absodepth=max_depth
    result=minimaxabc(game,max_depth,absodepth,player,playerAI,-math.inf,math.inf)
    newresult = result[0]
    depth = result[2]
    for _ in range(max_depth-depth-1):
        newresult = newresult.parent
    return newresult

def minimaxabc(game, max_depth,absodepth, player, playerAI, alpha, beta):
    game.createChildren(player)
    if max_depth==0 or game.children == []:
        values = (1.0, 0.4, 0.7, 0.4)
        board = cp.deepcopy(game.board)
        #print(player,'pre evaluate')
        value=(game,evaluatePlay_minmax(game,board,game.parentPlay,game.parentCell,player,values), max_depth)
        #print(value[0].board,' ',value[1],'depth ',max_depth)
        return value

    if player == 3-playerAI:
        value =(GameState([]), math.inf,absodepth)
        for state in game.children:
            evaluation = minimaxabc(state, max_depth - 1,absodepth, 3-player, playerAI, alpha, beta)
            #print(evaluation[0].board,' ',evaluation[1],'minimizer maxdepth %d' % max_depth)
            if evaluation[1]<value[1]:
                value=evaluation
            beta = min(beta, evaluation[1])
            if beta <= alpha:
                break
        return value

    value =(GameState([]), -math.inf,absodepth)
    for state in game.children:
        evaluation = minimaxabc(state, max_depth - 1,absodepth, 3-player, playerAI, alpha, beta)
        #print(evaluation[0].board,' ',evaluation[1], 'maximizer maxdepth %d' % max_depth)
        if evaluation[1]>value[1]:
            value=evaluation
        alpha = max(alpha, evaluation[1])
        if beta <= alpha:
            break
    return value

#i=y and j=x : tuples are (y,x)
def get_moves(game,cell):
    vect = [(1,0),(2,0),(1,1),(2,2),(1,-1),(2,-2),(-1,0),(-2,0),(-1,1),(-2,-2),(0,1),(0,2),(0,-1),(0,-2),(-1,-1),(-2,2)]
    #moves é um dicionario onde a chave de cada elemento é uma lista com a validade do mov (True/False) no indice 0 e o tipo de movimento no indice 1
    moves={}
    for mov in vect:
        play=(cell[0]+mov[0],cell[1]+mov[1])
        if play[0]<0 or play[0]>len(game.board)-1 or play[1]<0 or play[1]>len(game.board)-1 or game.board[play[1]][play[0]]!=0:
            moves[mov]=[False]
        else:
            moves[mov]=[True]
        
        if 1 in mov or -1 in mov:
            moves[mov].append(1)
        elif 2 in mov or -2 in mov:
            moves[mov].append(2)
    return moves

#draws the board on screen
def drawBoard(game, screen):
    n = len(game.board)
    screen.fill((255,255,255)) #background branco

    #desenha frame do tabuleiro
    pygame.draw.line(screen, (0,0,0), (0,0), (800,0), 2)
    pygame.draw.line(screen, (0,0,0), (0,0), (0,800), 2)
    pygame.draw.line(screen, (0,0,0), (0,798), (800,798), 2)
    pygame.draw.line(screen, (0,0,0), (798, 0), (798,800), 2)

    #desenha linhas do tabuleiro
    for i in range(1,n):
        #linhas verticais
        pygame.draw.line(screen, (0,0,0), (800*i/n,0), (800*i/n,800), 2)
        #linhas horizontais
        pygame.draw.line(screen, (0,0,0), (0,800*i/n), (800,800*i/n), 2)

def drawPieces(game, screen):
    n = len(game.board)
    for i in range(n):
        for j in range(n):
            #desenha peças do jogador 1
            if game.board[j][i] == 1:
                pygame.draw.circle(screen, (0,0,255), ((800*i/n)+800/(2*n), (800*j/n)+800/(2*n)), 800/(3*n))
            #desenha peças do jogador 2
            if game.board[j][i] == 2:
                pygame.draw.circle(screen, (0,150,0), ((800*i/n)+800/(2*n), (800*j/n)+800/(2*n)), 800/(3*n))
            #desenha quadrados onde não se pode jogar
            if game.board[j][i] == 8:
                pygame.draw.rect(screen, (0,0,0), (800*i/n, 800*j/n, 800/n + 1, 800/n + 1))

#mostrar o resultado do jogo graficamente
def drawResult(game, screen):
    if game.end == -1:
        return None
    font = pygame.font.Font('freesansbold.ttf', 32)
    pygame.draw.rect(screen, (0,0,0), (120, 240, 560, 320))
    pygame.draw.rect(screen, (255,255,255), (140, 260, 520, 280))
    if game.end == 0:
        text = font.render("Empate!", True, (0,0,0))
    elif game.end == 1:
        text = font.render("Jogador 1 vence!", True, (0,0,255))
    elif game.end == 2:
        text = font.render("Jogador 2 vence!", True, (0,150,0))
    text_rect = text.get_rect(center=(400, 400))
    screen.blit(text, text_rect)

#return the coordinates of the mouse in the game window
def mousePos(game):
    click = pygame.mouse.get_pos()   
    n = len(game.board)
    i = int(click[0]*n/800)
    j = int(click[1]*n/800)
    coord=(i,j)
    return coord

#shows the selected cell on screen
def showSelected(game, screen, coord, player_id):
    n = len(game.board)
    i=coord[0]
    j=coord[1]
    #selectedType é um dicionario onde cada elemento é um dos quadrados onde se pode jogar e cuja chave é o tipo de movimento
    selectedType = {}
    if game.board[j][i] == player_id:
        #desenha as cell possiveis de se jogar do player id
        if player_id == 1:
            selectedCellRGB  = (173,216,230) #azul claro
        elif player_id == 2:
            selectedCellRGB = (144,238,144) #verde claro
        pygame.draw.rect(screen, selectedCellRGB, (800*i/n + 2, 800*j/n + 2, 800/n - 2 , 800/n - 2))
        moves=get_moves(game,coord)
        for mov in moves:
            if moves[mov][0]:
                play=(coord[0]+mov[0],coord[1]+mov[1])
                selectedType[play] = moves[mov][1]
                pygame.draw.rect(screen, selectedCellRGB, (800*play[0]/n + 2, 800*play[1]/n + 2, 800/n - 2 , 800/n - 2))
    return selectedType

def get_and_apply_adjacent(targetCell, newBoard, player_id):
    vectors = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
    #adjacent é um dicionario que vai ter como elemento cada cell que esta a volta da targetCell e cujos elementos sao True/False
    #se essa cell tiver/não tiver uma peça do oponente
    adjacent = {}
    for vect in vectors:
        play=(targetCell[0]+vect[0],targetCell[1]+vect[1])
        if play[0]<0 or play[0]>len(newBoard)-1 or play[1]<0 or play[1]>len(newBoard)-1 or newBoard[play[1]][play[0]] != switchPlayer(player_id):
            adjacent[vect] = False
        else:
            adjacent[vect] = True
    for adj in adjacent:
        if adjacent[adj]:
            adjCell = (targetCell[0]+adj[0], targetCell[1]+adj[1])
            newBoard[adjCell[1]][adjCell[0]] = player_id
    return newBoard

def skip(game,player):
    game.createChildren(player)
    if len(game.children) == 0:
        return True
    return False

def objective_testmcts(game,player): #atualizar count
    gamenp=np.array(game.board)
    if np.count_nonzero(gamenp==0)==0:
        if np.count_nonzero(gamenp==player)>np.count_nonzero(gamenp==(3-player)):
            return player
        if np.count_nonzero(gamenp==player)<np.count_nonzero(gamenp==(3-player)):
            return 3-player
        else:
            return 0
    if np.count_nonzero(gamenp==player)==0:
        return (3-player+10)
    for j in range(len(gamenp)):
        for i in range(len(gamenp)):
            if gamenp[j][i]==player:
                if True in get_moves(game,(i,j)):
                    return -1
                else:
                    return (3-player+10)

def objective_test(game,player): #atualizar count
    gamenp=np.array(game.board)
    if np.count_nonzero(gamenp==(3-player))==0:
        return player
    if np.count_nonzero(gamenp==0) != 0:
                return -1
    if np.count_nonzero(gamenp==0) == 0:  
        count_p=np.count_nonzero(gamenp==player)
        count_o=np.count_nonzero(gamenp==(3-player))
        if count_p > count_o:
            return player
        if count_o > count_p:
            return (3-player)
    return 0 

def executeMov(game, initialCell, targetCell, selectedType, player_id):
    newBoard = cp.deepcopy(game.board)
    if targetCell in selectedType:
        movType = selectedType[targetCell]
        #movimento tipo 1
        if movType == 1:
            newBoard[targetCell[1]][targetCell[0]] = player_id
            newBoard = get_and_apply_adjacent(targetCell, newBoard, player_id)
        #movimento tipo 2
        elif movType == 2:
            newBoard[targetCell[1]][targetCell[0]] = player_id
            newBoard[initialCell[1]][initialCell[0]] = 0
            newBoard = get_and_apply_adjacent(targetCell, newBoard, player_id)
    newGame = GameState(newBoard)
    return newGame

def switchPlayer(player_id):
    return 3-player_id

#game mode Human vs Human
def jogo_Humano_Humano(game, screen):
    player_id = 1
    clickState = False
    while game.end==-1:
        drawPieces(game, screen)
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

            #verificar se o jogador está cercado / não tem jogadas possiveis e tem de passar a jogada
            if not skip(game,player_id):
                #escolher a peca para jogar e as possiveis plays
                if event.type == pygame.MOUSEBUTTONDOWN and clickState == False:
                    drawBoard(game, screen)
                    coord = mousePos(game)
                    selected = showSelected(game, screen, coord, player_id)
                    clickState = True
                    drawPieces(game, screen)

                #fazer o movimento da jogada
                elif event.type == pygame.MOUSEBUTTONDOWN and clickState == True:
                    targetCell = mousePos(game)
                    prevBoard = cp.deepcopy(game.board)
                    game = executeMov(game, coord, targetCell, selected, player_id)
                    if not (np.array_equal(prevBoard,game.board)):
                        player_id = switchPlayer(player_id)
                    clickState=False
                    drawBoard(game, screen)
                    drawPieces(game, screen)
            else:
                player_id = switchPlayer(player_id)
        game.end = objective_test(game,player_id)

        #to display the winner
        while game.end != -1:
            drawResult(game,screen)
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
            pygame.display.update()
        pygame.display.update()

def jogo_Humano_AI(game, screen, algorithm):
    player_id = 1
    clickState = False
    while game.end==-1:
        drawPieces(game, screen)
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

            #verificar se o jogador está cercado / não tem jogadas possiveis e tem de passar a jogada
            if not skip(game,player_id):
                if player_id == 1:
                    #escolher a peca para jogar e as possiveis plays
                    if event.type == pygame.MOUSEBUTTONDOWN and clickState == False:
                        drawBoard(game, screen)
                        coord = mousePos(game)
                        selected = showSelected(game, screen, coord, player_id)
                        clickState = True
                        drawPieces(game, screen)

                    #fazer o movimento da jogada
                    elif event.type == pygame.MOUSEBUTTONDOWN and clickState == True:
                        targetCell = mousePos(game)
                        prevBoard = cp.deepcopy(game.board)
                        game = executeMov(game, coord, targetCell, selected, player_id)
                        if not (np.array_equal(prevBoard,game.board)):
                            player_id = switchPlayer(player_id)
                        clickState=False
                        drawBoard(game, screen)
                        drawPieces(game, screen)
                else:
                    if algorithm == 1:
                        game = implement_minimax(game,player_id, player_id)
                    elif algorithm == 2:
                        game = implementar_montecarlos(game,player_id)
                    elif algorithm == 3:
                        game = greedy(game,player_id)
                    elif algorithm == 4:
                        game = randomplay(game, player_id)
                    drawBoard(game, screen)
                    drawPieces(game, screen)
                    player_id = 1
            else:
                player_id = switchPlayer(player_id)
        game.end = objective_test(game,player_id)

        #to display the winner
        while game.end != -1:
            drawResult(game,screen)
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
            pygame.display.update()
        pygame.display.update()

#sets the game window
def setScreen():
    width = 800
    height = 800
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Ataxx")
    return screen

#abre um ficheiro com um mapa do tabuleiro a ser usado no jogo e cria o estado/objeto inicial
def readBoard(ficheiro):
    f = open(ficheiro, "r")
    n = int(f.readline())
    board = []
    for i in range(n):
        board.append(list(map(int, f.readline().split())))
    f.close()
    return GameState(board)

#pede ao user para escolher o tabuleiro que pretende usar
def chooseBoard():
    #todos os ficheiros com tabuleiros devem ter nome do tipo "tabX.txt"
    tableNum = input("Escolha o número do tabuleiro que quer usar para o jogo!\n1) 10x10\n2) 8x8\n3) 6x6\n4) 5x5\n5) 12x12\nTabuleiro: ")
    table = "tab"+tableNum+".txt"
    return table

def chooseMode():
    mode = int(input("Escolha o modo de jogo!\n1) Humano vs Humano\n2) Humano vs AI\nModo: "))
    return mode

def chooseAI():
    algorithm=int(input("Escolha o seu adversário!\n1) Minimax\n2) MonteCarloTreeSearch **VT**\n3) Greedy\n4) Random Play\nModo: "))
    return algorithm

def playMode(game, screen, mode,algorithm):
    if mode == 1:
        jogo_Humano_Humano(game, screen)
    elif mode == 2:
        jogo_Humano_AI(game,screen,algorithm)

def simulacao(numSimulations):
    playerTurns = [1,2,1,1]
    empate = 0
    w1 = 0
    comeutodas1 = 0
    w2 = 0
    comeutodas2 = 0
    for i in range(numSimulations):
        table = "tabSim" + str(i+1) + ".txt"
        #table = "tabSim18.txt"
        game = readBoard(table)
        #player_id = playerTurns[i%4]
        player_id = 1
        while game.end == -1:
            if not skip(game,player_id):
                if player_id == 1:
                    game = greedy(game,player_id)  #para testar os algoritmos da AI é só trocar aqui a função pelo algoritmo desejado
                    player_id = switchPlayer(player_id)
                elif player_id == 2:
                    game = implement_minimax(game, player_id, player_id)  #igual ao comentario acima
                    player_id = switchPlayer(player_id)
            else:
                player_id = switchPlayer(player_id)
            game.end = objective_test(game, player_id)

        print(i+1)
        if game.end == 0:
            empate += 1
        elif game.end == 1:
            w1 += 1
            gamenp = np.array(game.board)
            if np.count_nonzero(gamenp == 2) == 0:
                comeutodas1 += 1
        elif game.end == 2:
            w2 += 1
            gamenp = np.array(game.board)
            if np.count_nonzero(gamenp == 1) == 0:
                comeutodas2 += 1
    
    with open("data.txt", "a") as data:
        numbers = "%d %d %d %d\n" % (numSimulations, w1, w2, empate)
        data.write(numbers)
    print("Jogos: %d\nAI 1: %d (Jogos com todas as peças do oponente eliminadas: %d)\nAI 2: %d (Jogos com todas as peças do oponente eliminadas: %d)\nEmpate: %d\n" % (numSimulations, w1, comeutodas1, w2, comeutodas2, empate))

def main():
    mode = chooseMode()
    if mode==2:
        algorithm=chooseAI()
    else:
        algorithm=0
    table = chooseBoard()
    pygame.init()
    screen = setScreen()
    game = readBoard(table)
    drawBoard(game, screen)
    playMode(game, screen, mode,algorithm)

start_time = time.time()
main()
#simulacao(100)
print("--- %.5f seconds ---" % (time.time() - start_time))