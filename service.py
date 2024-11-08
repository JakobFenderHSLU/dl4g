# HSLU
#
# Created by Thomas Koller on 7/30/2020
# Modified by Dan Livingston & Jakob Fender
#
import json
import logging
import os

from flask import jsonify, request
from jass.game.game_observation import GameObservation
from jass.service.player_service_app import PlayerServiceApp

from src.agent.agent import CustomAgent
from src.play_strategy.nn.mcts.dmcts_worker import DMCTSWorker
from src.play_strategy.random_play_strategy import RandomPlayStrategy
from src.trump_strategy.random_trump_strategy import RandomTrumpStrategy

dmcts_worker = DMCTSWorker(os.getenv("LIMIT_S", 1.0))  # TODO: set realistic limit_s


def create_app():
    logging.basicConfig(level=logging.INFO)

    # create and configure the app
    app = PlayerServiceApp("player_service")

    # add some players
    app.add_player(
        "random",
        CustomAgent(
            trump_strategy=RandomTrumpStrategy(0),
            play_strategy=RandomPlayStrategy(),
            play_rules_strategies=[],
        ),
    )

    return app


def modify_app(app):
    @app.route("/ping", methods=["GET"])
    @app.route("/ping", methods=["POST"])
    def ping():
        return jsonify("pong")

    # http://127.0.0.1:5000/dmcts?obs={%22version%22:%20%22V0.2%22,%20%22trump%22:%201,%20%22dealer%22:%200,%20%22currentPlayer%22:%203,%20%22playerView%22:%203,%20%22forehand%22:%201,%20%22tricks%22:%20[{%22first%22:%203}],%20%22player%22:%20[{%22hand%22:%20[]},%20{%22hand%22:%20[]},%20{%22hand%22:%20[]},%20{%22hand%22:%20[%22D8%22,%20%22D7%22,%20%22HK%22,%20%22H9%22,%20%22SA%22,%20%22SQ%22,%20%22S10%22,%20%22S7%22,%20%22CK%22]}],%20%22jassTyp%22:%20%22SCHIEBER%22}
    @app.route("/dmcts", methods=["GET"])
    def dmcts():
        obs_str: str = request.args.get("obs", None)
        if obs_str is None:
            return jsonify("No observation provided"), 422

        obs_json = json.loads(obs_str)
        obs = GameObservation.from_json(obs_json)
        action_values = dmcts_worker.execute(obs)
        return jsonify(action_values.tolist())

    return app


if __name__ == "__main__":
    app = create_app()
    app = modify_app(app)
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 5000))

    app.run(host=host, port=port)
