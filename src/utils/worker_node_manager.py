import asyncio
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
            self.worker_nodes: list[WorkerNode] = []
            self.load_worker_nodes(file_path)
            logging.info(f"loaded {len(self.worker_nodes)} node(s)")
            logging.info(f"Worker nodes: {[node.name for node in self.worker_nodes]}")
            asyncio.run(self.ping_nodes_remove_failed())
            logging.info(f"kept {len(self.worker_nodes)} node(s)")
            logging.info(f"Worker nodes: {[node.name for node in self.worker_nodes]}")
        else:
            logging.info("Returning existing instance of WorkerNodeManager singleton")

    def load_worker_nodes(self, file_path: str) -> None:
        """Load worker nodes from a JSON file and add them to a max-heap."""

        with open(file_path, "r") as file:
            data = json.load(file)
        for node in data["nodes"]:
            if node["enabled"]:
                worker_node = WorkerNode(node["name"], node["ip"], node["port"])
                self.worker_nodes.append(worker_node)

    async def ping_nodes_remove_failed(self) -> None:
        """Pings all worker nodes and removes those that fail to respond."""

        temp_nodes = []
        tasks = [
            (worker_node, asyncio.create_task(worker_node.ping()))
            for worker_node in self.worker_nodes
        ]
        for worker_node, task in tasks:
            try:
                if await task:
                    temp_nodes.append(worker_node)
            except asyncio.TimeoutError:
                logging.warning(f"Timeout while pinging {worker_node.name}")
            except Exception as e:
                logging.error(f"Error while pinging {worker_node.name}: {e}")
        self.worker_nodes = temp_nodes

    async def process_game_observation(self, worker_node, obs_json):
        return await worker_node.process_game_observation(obs_json)

    def execute_all_dmcts(self, obs_json):
        logging.info("Executing DMCTS on nodes")
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            logging.debug("No current event loop, creating a new one.")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        tasks = [
            self.process_game_observation(worker_node, obs_json)
            for worker_node in self.worker_nodes
        ]
        results = loop.run_until_complete(asyncio.gather(*tasks))
        logging.debug("results")
        logging.debug(results)
        return [result for result in results if result is not None]
