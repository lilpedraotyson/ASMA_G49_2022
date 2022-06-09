import gym
import numpy as np
import matplotlib.pyplot as plt
from time import sleep

from randomGhost import RandomGhost

from ghost import Ghost

from environment import Environment

def showPieResults(agent, win_count):
    #labels = ['win', 'loss']
    fig1, ax1 = plt.subplots()
    ax1.pie([100 - win_count, win_count], labels=['win', 'loss'], explode= (0.1, 0), autopct='%1.1f%%', shadow=True, startangle=90)
    ax1.axis('equal')
    ax1.set_title(agent)

def showResults(agent, n_steps):
    fig, ax = plt.subplots()
    x = np.arange(len(agent))
    rects1 = ax.bar(x, n_steps, label='average step counter', align='center', color='orange')

    ax.set_ylabel('Number of Steps')
    ax.set_title('Number of steps per agent')
    ax.set_xticks(x, agent)
    ax.legend()

    ax.bar_label(rects1, padding=3)

    fig.tight_layout()

    plt.show()

def run_multiple_agents(environment, n_agents, episodes, tipo, alive=True):
    win_count = 0
    if tipo == "random":
        environment.ghosts = {i: RandomGhost(i, environment.n_ghosts) for i in range(environment.n_ghosts)}
    else:
        environment.ghosts = {i: Ghost(i, environment.n_ghosts) for i in range(environment.n_ghosts)}
    observation = environment.reset()
    print('ghost position: {} pacman position: {}'.format(observation[:(n_agents - 1) * 2], observation[(n_agents - 1) * 2:]))
    
    #sleep(0.5)
    meanaux = []
    for _ in range(episodes):
        print("----------------------------------------------------------------------------------------------------")
        environment.reset()
        alive = environment.pacman.alive
        while alive and environment.step_count < 200 and [environment.ghosts[i].alive for i in range(n_agents - 1)] == [True for i in range(n_agents - 1)]:
            for i in range(n_agents - 1):
                environment.ghosts[i].see(observation)
                environment.pacman.see(observation)
            next_obs, alive = environment.step([environment.ghosts[i].action(environment.map, environment.step_count) for i in range(n_agents - 1)] + [environment.pacman.action(environment.step_count)])
            print('positions: {} pacman alive?: {}'.format(next_obs, alive))
            observation = next_obs
            environment.render()
            #sleep(0.5)
        meanaux.append(environment.step_count)
        if not alive:
            win_count += 1
    print(meanaux)
    return sum(meanaux) / len(meanaux), int((win_count/episodes) * 100)

if __name__ == '__main__':
    n_agents = 4
    environment = Environment(n_agents - 1)
    episodes = 10
    tipo = ["random", "hybrid"]
    mean = []

    #aux_mean, aux_win = run_multiple_agents(environment, n_agents, episodes, tipo[0])
    #print(aux_mean)
    #print(aux_win)
    #mean.append(aux_mean)
    #showPieResults(tipo[0], aux_win)

    aux_mean, aux_win = run_multiple_agents(environment, n_agents, episodes, tipo[1])
    print(aux_mean)
    print(aux_win)
    mean.append(aux_mean)
    showPieResults(tipo[1], aux_win)
    showResults([tipo[1]], mean)