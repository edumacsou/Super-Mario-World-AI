# Código escrito para a disciplina de Inteligencia Artificial, ministrada pelo professor Fabrício Olivetti de França
# Esse código possui trechos extraídos de https://github.com/Sentdex/NEAT-samples

from rominfo import *
from utils import *

import numpy as np
import multiprocessing      # Atividades paralelas
import pickle               # Salvar e carregar arquivos
import os, sys              # Interagir com o terminal
import neat                 # Rede Neural
import retro                # Emula o jogo


show = True                # Define se o o treinamento será visível(lento) ou não(rápido) 
raio = 6                    # Distância de captação ao redor do Mario

def play_the_game(genome, config):
    ''' Roda uma fase do jogo usando uma rede de classificação com os parâmetros recebidos pelo genoma. Retorna o valor do progresso na fase '''

    net = neat.nn.FeedForwardNetwork.create(genome, config)
    moves = {'direita':128,                 # Apenas esses movimentos são necessários para passar a fase
             'corre':130,
             'pula':131,
             'spin':386,
             'esquerda':64 }


    # Carregamento das informações do jogo
    env = retro.make(game='SuperMarioWorld-Snes', state='YoshiIsland2', players=1)
    env.reset()
    estado, xn, y = getState(getRam(env), raio)


    fitness = 0.0               # Progresso alcançado
    done = False                # Indica o final do Jogo
    x_max = 0                   # Auxiliar para identificar se o Mario regrediu na fase (andou pra trás)
    contador_retorno = 0        # Armazena quantas ações foram realizadas sem avançar na fase (retornar no caminho)

    while not done:
        # Carrega informações do estado atual
        estado, xn, y = getState(getRam(env), raio)
        state_n = np.reshape(estado.split(','), (2*raio + 1, 2*raio + 1))
        
        # Guarda o estado em uma lista de inteiros (padrão de inputs do NEAT)
        estado_tratado = []
        for line in state_n:
            for num in line:
                estado_tratado.append(int(num))

        # Insere os Inputs na rede de classificação e recebe uma lista de Outputs
        actions = net.activate(estado_tratado)

        # Atribui uma ação para cada um dos valores de Output
        actions = [(actions[i], list(moves.items())[i][0])
                    for i in range(len(list(moves.items())))]

        # Escolhe a ação com maior valor
        action = sorted(actions, key=lambda x:x[0], reverse=True)[0]

        # Transforma o ambiente executando a ação escolhida
        reward, done = performAction(moves[action[1]], env)

        # Verifica se o Mario ficou estagnado ou voltou para trás.
        if xn>x_max:
            x_max = xn
            contador_retorno = 0
        else:
            contador_retorno += 1
        
        # Termina o jogo caso o Mario fique muito tempo sem se mover ou progredir
        if contador_retorno == 100:
            done = True


        fitness = x_max*10              # Aumentar o fitness em 10 vezes ajudou no processo de classificar os melhores da geração
        
        global show
        if show:
            env.render()
    
    # Fecha o ambiente, para não ocupar a memória
    env.render(close=True)
    env.close()
    

    return np.mean(fitness)         # Devolve uma npArray, formato usado pela biblioteca NEAT. A Função mean não altera o valor de fitness por ele ser único.

def eval_genomes(genomes, config):
    ''' Testa todos os indivíduos gerados e salva o progresso em seu genoma '''

    for genome_id, genome in genomes:
        genome.fitness = play_the_game(genome, config)

def train():
    ''' Treina a IA até completar a fase '''

    # Carrega os dados no arquivo de configurações
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config')
    config = neat.Config(neat.DefaultGenome,
                         neat.DefaultReproduction,
                         neat.DefaultSpeciesSet,
                         neat.DefaultStagnation,
                         config_path)

    # Inicia a população Inicial
    pop = neat.Population(config)

    # Inicia o Reporter das estatísticas, que são impressas a cada geração e quando se encontra o vencedor
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))

    # Declara que a função de avaliação dos indivíduos será feita em paralela, segundo a quantidade de 'Cores' disponíveis
    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), play_the_game)
    
    # Inicia a busca pelo ganhador
    # run não é a função declarada abaixo e sim um método da classe Population
    winner = pop.run(pe.evaluate)

    # Salva o ganhador.
    with open('winner', 'wb') as f:
        pickle.dump(winner, f)

def run(winner_path="winner"):
    ''' Carrega um indivíduo ganhador e joga a fase inteira '''

    # Carrega os arquivos de configuração
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config')
    winner_path = os.path.join(local_dir, 'winner')
    config = neat.Config(neat.DefaultGenome,
                         neat.DefaultReproduction,
                         neat.DefaultSpeciesSet,
                         neat.DefaultStagnation,
                         config_path)
    
    # Tratamento da excessão caso não exista um arquivo de ganhador
    try:
        with open(winner_path, "rb") as f:
            genome = pickle.load(f)
    except FileNotFoundError:
        print("Não Existe um arquivo de gene válido.")
        print("Caminho fornecido:", winner_path)
        exit()

    global show
    show = True
    play_the_game(genome, config)


if __name__ == "__main__":
    ''' Faz o tratamento das entradas e chama as funções para treino ou execução '''

    if len(sys.argv) < 2:
        print("Escolha uma opção entre run e train para executar o programa")
        exit()
    if sys.argv[1] == 'train':
        train()
    elif sys.argv[1] == 'run':
        run()
    else:
        print("A opção digitada não é válida, favor escolher 'train' ou 'run'")
