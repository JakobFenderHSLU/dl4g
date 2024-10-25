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


def create_app():
    logging.basicConfig(level=logging.DEBUG)

    # create and configure the app
    app = PlayerServiceApp("player_service")

    # add some players
    app.add_player("random", AgentRandomSchieber())
    # app.add_player(
    #     "random",
    #     CustomAgent(
    #         trump_strategy=RandomTrumpStrategy(0),
    #         play_strategy=RandomPlayStrategy(),
    #         play_rules_strategies=PlayRuleStrategy(),
    #     ),
    # )

    return app


def modify_app(app):
    @app.route("/ping", methods=["GET"])
    @app.route("/ping", methods=["POST"])
    def ping():
        return jsonify("pong")

    return app


if __name__ == "__main__":
    app = create_app()
    app = modify_app(app)
    app.run(host="0.0.0.0", port=5000)
