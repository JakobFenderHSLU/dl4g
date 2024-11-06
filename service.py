# HSLU
#
# Created by Thomas Koller on 7/30/2020
# Modified by Dan Livingston & Jakob Fender
#
import logging

from flask import jsonify
from jass.agents.agent_random_schieber import AgentRandomSchieber
from jass.service.player_service_app import PlayerServiceApp

from src.agent.agent import CustomAgent
from src.play_rule_strategy.only_valid_play_strategy import PlayRuleStrategy
from src.play_strategy.random_play_strategy import RandomPlayStrategy
from src.trump_strategy.random_trump_strategy import RandomTrumpStrategy
from src.utils.worker_nodes import WorkerNodeManager
import os


def create_app():
    logging.basicConfig(level=logging.INFO)

    # create and configure the app
    app = PlayerServiceApp("player_service")

    # add some players
    app.add_player("random", AgentRandomSchieber())
    # app.add_player(
    #     "random",
    #     CustomAgent(
    #         trump_strategy=RandomTrumpStrategy(0),
    #         play_strategy=RandomPlayStrategy(),
    #         play_rules_strategies=[],
    #     ),
    # )

    return app


def modify_app(app):
    @app.route("/ping", methods=["GET"])
    @app.route("/ping", methods=["POST"])
    def ping():
        return jsonify("pong")

    @app.route("/process", methods=["GET"])
    def process_obs():
        # TODO: process game observation
        # TODO: return result
        return jsonify([])

    return app


if __name__ == "__main__":
    app = create_app()
    app = modify_app(app)
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 5000))

    app.run(host=host, port=port)
