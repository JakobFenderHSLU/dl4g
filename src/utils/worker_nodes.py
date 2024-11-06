import heapq
import json
import logging
import os

from src.utils.worker_node import WorkerNode


class WorkerNodeManager:
    """WorkerNodeManager is a singleton class that manages a collection of WorkerNode instances."""

    _instance = None

    def __new__(cls, *args, **kwargs):
        """Create a new instance of WorkerNodeManager if one does not exist, ensuring a singleton pattern."""

        if cls._instance is None:
            cls._instance = super(WorkerNodeManager, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self) -> None:
        if not hasattr(self, "initialized"):  # Ensure __init__ is called only once
            self.initialized = True
            logging.info("Instantiating WorkerNodeManager singleton")
            file_path = (
                f"{os.path.dirname(os.path.realpath(__file__))}/worker_nodes.json"
            )
            self.worker_nodes = []
            self.load_worker_nodes(file_path)
            logging.info(f"loaded {len(self.worker_nodes)} node(s)")
            self.ping_nodes_remove_failed()
            logging.info(f"kept {len(self.worker_nodes)} node(s)")
        else:
            logging.info("Returning existing instance of WorkerNodeManager singleton")

    def load_worker_nodes(self, file_path: str) -> None:
        """Load worker nodes from a JSON file and add them to a max-heap."""

        with open(file_path, "r") as file:
            data = json.load(file)
        for node in data["nodes"]:
            if node["enabled"]:
                worker_node = WorkerNode(node["name"], node["ip"], node["port"])
                heapq.heappush(
                    self.worker_nodes, (-node["cpu_cores"], worker_node)
                )  # Use negative to create a max-heap

    def flush_worker_nodes(self) -> None:
        """Clears the list of worker nodes."""

        self.worker_nodes = []

    def ping_nodes_remove_failed(self) -> None:
        """Pings all worker nodes and removes those that fail to respond."""

        temp_nodes = []
        while self.worker_nodes:
            cpu_cores, worker_node = heapq.heappop(self.worker_nodes)
            if worker_node.ping():
                heapq.heappush(temp_nodes, (cpu_cores, worker_node))
        self.worker_nodes = temp_nodes

    def get_available_node(self) -> WorkerNode:
        """Retrieves and returns the WorkerNode instance with the highest CPU cores available."""
        if self.worker_nodes:
            return heapq.heappop(self.worker_nodes)[1]  # Return the WorkerNode instance
        return None

    def return_available_node(self, node: WorkerNode) -> None:
        """Adds a worker node (back) to the heap of available nodes, prioritizing by CPU cores."""

        heapq.heappush(self.worker_nodes, (-node.cpu_cores, node))
