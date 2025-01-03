# HSLU
#
# Created by Thomas Koller on 7/30/2020
# Modified by Dan Livingston & Jakob Fender
#
import json
import logging
import os
import threading
import time

import psutil
from flask import jsonify, request
from jass.game.game_observation import GameObservation
from jass.service.player_service_app import PlayerServiceApp

from src.agent.agent import CustomAgent
from src.play_rule_strategy.only_valid_play_strategy import OnlyValidPlayRuleStrategy
from src.play_rule_strategy.smear_play_strategy import SmearPlayRuleStrategy
from src.play_rule_strategy.trump_jack_strategy import TrumpJackPlayRuleStrategy
from src.play_strategy.determinized_mcts_play_strategy import (
    DeterminizedMCTSPlayStrategy,
)
from src.play_strategy.nn.mcts.dmcts_worker import DMCTSWorker
from src.play_strategy.random_play_strategy import RandomPlayStrategy
from src.trump_strategy.deep_nn_trump_strategy import DeepNNTrumpStrategy
from src.trump_strategy.random_trump_strategy import RandomTrumpStrategy
from src.utils.worker_node_manager import WorkerNodeManager


def create_app():
    logging.info("Creating App")
    # create and configure the app
    app = PlayerServiceApp("player_service")
    seed = os.getenv("SEED", 42)
    log_level = os.getenv("LOG_LEVEL", "INFO")

    # add some players
    app.add_player(
        "random",
        CustomAgent(
            trump_strategy=RandomTrumpStrategy(0),
            play_strategy=RandomPlayStrategy(),
            play_rules_strategies=[],
        ),
    )
    app.add_player(
        "dmcts-player",
        CustomAgent(
            trump_strategy=DeepNNTrumpStrategy(),
            play_strategy=DeterminizedMCTSPlayStrategy(
                limit_s=os.getenv("LIMIT_S", 9.0)
            ),
            play_rules_strategies=[
                OnlyValidPlayRuleStrategy(seed=seed, log_level=log_level),
                SmearPlayRuleStrategy(seed=seed, log_level=log_level),
                TrumpJackPlayRuleStrategy(seed=seed, log_level=log_level),
            ],
        ),
    )

    return app


def modify_app(app):
    logging.info("Modifying App")

    dmcts_worker = DMCTSWorker(
        limit_s=float(os.getenv("LIMIT_S", 8.0)),
        n_determinations=int(os.getenv("N_DETERMINATIONS", psutil.cpu_count(logical=False))),
        n_iterations=int(os.getenv("N_ITERATIONS")) if os.getenv("N_ITERATIONS") else None,
    )

    @app.route("/ping", methods=["GET"])
    @app.route("/ping", methods=["POST"])
    def ping():
        return jsonify("pong")

    @app.route("/reload-worker-nodes")
    def reload_worker_nodes():
        w = WorkerNodeManager()
        w.reload_all_worker_nodes()
        return jsonify(w.get_worker_nodes_dict())

    # http://127.0.0.1:5000/dmcts?obs={%22version%22:%20%22V0.2%22,%20%22trump%22:%201,%20%22dealer%22:%200,%20%22currentPlayer%22:%203,%20%22playerView%22:%203,%20%22forehand%22:%201,%20%22tricks%22:%20[{%22first%22:%203}],%20%22player%22:%20[{%22hand%22:%20[]},%20{%22hand%22:%20[]},%20{%22hand%22:%20[]},%20{%22hand%22:%20[%22D8%22,%20%22D7%22,%20%22HK%22,%20%22H9%22,%20%22SA%22,%20%22SQ%22,%20%22S10%22,%20%22S7%22,%20%22CK%22]}],%20%22jassTyp%22:%20%22SCHIEBER%22}
    @app.route("/dmcts", methods=["GET"])
    def dmcts():
        start_time = time.time()
        obs_str: str = request.args.get("obs", None)
        if obs_str is None:
            return jsonify("No observation provided"), 422

        obs_json = json.loads(obs_str)
        obs = GameObservation.from_json(obs_json)
        action_values = dmcts_worker.execute(obs)

        execution_time = time.time() - start_time
        if execution_time > 9.5:
            logging.error(
                f"Execution time /dmcts exceeded 9.5 seconds: {execution_time:.2f} seconds"
            )
        else:
            logging.info(f"Execution time /dmcts: {execution_time:.2f} seconds")

        return jsonify(action_values.tolist())

    return app


def delayed_worker_node_init():
    logging.debug("Delaying WorkerNodeManager initialization")
    time.sleep(3)
    WorkerNodeManager()
    logging.debug("WorkerNodeManager initialized")


if __name__ == "__main__":
    # attempt to load .env
    try:
        from dotenv import load_dotenv

        load_dotenv()
        print("Loaded .env")
    except ImportError:
        print("Skipped .env due to ImportError")
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

    app = create_app()
    app = modify_app(app)
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 5000))

    threading.Thread(target=delayed_worker_node_init).start()

    app.run(host=host, port=port)
