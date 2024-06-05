import torch
import matplotlib.pyplot as plt

class HebbianHopfieldNetwork:
    def __init__(self, n_units):
        self.n_units = n_units
        self.weights = torch.zeros((n_units, n_units), dtype=torch.float32)

    def train(self, patterns):
        self.weights.zero_()
        for p in patterns:
            p = p.view(-1, 1)
            self.weights += torch.mm(p, p.t())
        self.weights.fill_diagonal_(0)

    def recall(self, pattern, steps=5):
        state = pattern.clone()
        energy_values = []
        for step in range(steps):
            for i in range(self.n_units):
                net_input = torch.dot(self.weights[i], state)
                state[i] = torch.sign(net_input)
            energy_values.append(self.energy(state).item())
            print(f"Step {step+1}, State: {state.tolist()[:5]}, Energy: {energy_values[-1]}")
        return state, energy_values

    def energy(self, state):
        return -0.5 * torch.sum(torch.mm(state.view(1, -1), torch.mm(self.weights, state.view(-1, 1))))

class CombinedHopfieldNetwork(HebbianHopfieldNetwork):
    def __init__(self, n_units, alpha=0.1):
        super().__init__(n_units)
        self.alpha = alpha  # learning rate for Rescorla-Wagner-like updates

    def rw_modulation(self, pattern, expected_output):
        prediction = torch.sign(torch.mv(self.weights, pattern))
        prediction_error = expected_output - prediction
        for i in range(self.n_units):
            for j in range(self.n_units):
                if i != j:
                    self.weights[i, j] += self.alpha * pattern[i] * prediction_error[j]

    def train(self, patterns, expected_outputs):
        super().train(patterns)
        for pattern, expected in zip(patterns, expected_outputs):
            self.rw_modulation(pattern, expected)

def generate_patterns(n_patterns, n_units):
    patterns = torch.randint(0, 2, (n_patterns, n_units)).float()
    patterns[patterns == 0] = -1
    return patterns

def evaluate_network(network, patterns):
    for pattern in patterns:
        recalled_pattern, _ = network.recall(pattern)
        if not torch.equal(recalled_pattern, pattern):
            return False
    return True

def plot_energy(energy_values, title):
    plt.figure()
    plt.plot(energy_values)
    plt.xlabel('Step')
    plt.ylabel('Energy')
    plt.title(title)
    plt.show()

# Function to find maximum memory capacity
def find_max_memory_capacity(n_units, max_patterns, alpha=0.1):
    max_capacity_hebbian = 0
    max_capacity_combined = 0

    for n_patterns in range(1, max_patterns + 1):
        patterns = generate_patterns(n_patterns, n_units)

        # Hebbian Network
        hebbian_net = HebbianHopfieldNetwork(n_units)
        hebbian_net.train(patterns)
        if evaluate_network(hebbian_net, patterns):
            max_capacity_hebbian = n_patterns
        else:
            break

        # Combined Network
        combined_net = CombinedHopfieldNetwork(n_units, alpha)
        combined_net.train(patterns, patterns)
        if evaluate_network(combined_net, patterns):
            max_capacity_combined = n_patterns
        else:
            break

    return max_capacity_hebbian, max_capacity_combined

# Example usage
if __name__ == "__main__":
    n_units = 100  # Number of units in the network
    max_patterns = 100  # Maximum number of patterns to test
    alpha = 0.1  # Learning rate for Rescorla-Wagner-like updates
    steps = 10  # Number of steps for recall

    max_capacity_hebbian, max_capacity_combined = find_max_memory_capacity(n_units, max_patterns, alpha)
    print("Max memory capacity of Hebbian Hopfield Network:", max_capacity_hebbian)
    print("Max memory capacity of Combined Hopfield Network:", max_capacity_combined)

    patterns = generate_patterns(10, n_units)

    # Hebbian Network
    hebbian_net = HebbianHopfieldNetwork(n_units)
    hebbian_net.train(patterns)
    for pattern in patterns:
        recalled_pattern, energy_values = hebbian_net.recall(pattern, steps)
        plot_energy(energy_values, f'Hebbian Network Energy (Pattern: {pattern.tolist()[:5]}...)')

    # Combined Network
    combined_net = CombinedHopfieldNetwork(n_units, alpha)
    combined_net.train(patterns, patterns)
    for pattern in patterns:
        recalled_pattern, energy_values = combined_net.recall(pattern, steps)
        plot_energy(energy_values, f'Combined Network Energy (Pattern: {pattern.tolist()[:5]}...)')
