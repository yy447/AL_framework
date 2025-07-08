class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.logits = []
        self.state_values = []
        self.is_terminals = []

    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.logits[:]
        del self.state_values[:]
        del self.is_terminals[:]

    def size(self):
        return len(self.rewards)
