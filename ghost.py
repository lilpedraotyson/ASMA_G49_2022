from agent import Agent

N_ACTIONS = 4
DOWN, LEFT, UP, RIGHT = range(N_ACTIONS)

class Ghost(Agent):
    def __init__(self, agent_id, n_agents):
        super(Ghost, self).__init__(f"Ghost")
        self.agent_id = agent_id
        self.n_agents = n_agents
        self.n_actions = N_ACTIONS
        self.position = [13, 11 + agent_id]
        
    def reset_position(self):
        self.position = [13, 11 + self.agent_id]

    def print_position(self):
        print (self.position)

    def get_position(self):
        return self.position

    def set_position(self, line, column):
        self.position = [line, column]

    def action(self) -> int:
        return 0