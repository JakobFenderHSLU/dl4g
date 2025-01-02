# Distributed DMCTS Node System

The work distribution for DMCTS (Determinized Monte Carlo Tree Search) across nodes is managed by the `WorkerNodeManager` class. Here's a step-by-step explanation of how it works:

1. **Initialization**:

- The `WorkerNodeManager` is a singleton class that ensures only one instance is created.
- During initialization, it loads worker nodes from a JSON file (`worker_nodes.json`) and pings them to ensure they are active.

2. **Loading Worker Nodes**:

- The `load_worker_nodes` method reads the `worker_nodes.json` file and creates `WorkerNode` instances for each enabled node.
- These nodes are stored in the `worker_nodes` list.

3. **Pinging Nodes**:

- The `ping_nodes_remove_failed` method asynchronously pings all worker nodes to check their availability.
- Nodes that do not respond are removed from the `worker_nodes` list.

4. **Executing DMCTS**:

- When the `execute_all_dmcts` method is called with a game observation in JSON format (`obs_json`), it distributes the work across all active worker nodes.
- It creates an asyncio event loop and schedules tasks to process the game observation on each worker node using the `process_game_observation` method.

5. **Processing Game Observation**:

- Each `WorkerNode` instance sends an HTTP GET request to its corresponding node's `/dmcts` endpoint with the game observation.
- The node processes the observation and returns the action values.

6. **Collecting Results**:

- The `execute_all_dmcts` method gathers the results from all nodes.
- It filters out any `None` results (indicating failed nodes) and returns the valid action scores.

## Ensuring a 10-Second Response from the Player

To ensure that the player responds within 10 seconds, the system employs several strategies:

### 1. Timeouts

Each worker node has a timeout of 9 seconds for processing game observations. This is implemented in the `WorkerNode` class using the `aiohttp` library. If a node takes longer than 9 seconds to respond, it is considered a failure, and its result is discarded.

### 2. Parallel Processing

The `WorkerNodeManager` distributes the work across multiple nodes in parallel using `asyncio`. This reduces the overall computation time by leveraging concurrent processing.

### 3. Execution Time Logging

Both the `choose_card` method in `DeterminizedMCTSPlayStrategy` and the `/dmcts` endpoint in `service.py` log the execution time. If the execution time exceeds 9.5 seconds, an error is logged.

### 4. Fallback Mechanism

In the `choose_card` method, if no action scores are returned or an error occurs, the first valid card is chosen as a fallback.

This distributed approach allows the DMCTS algorithm to leverage multiple nodes for parallel processing, improving efficiency and reducing the overall computation time.
