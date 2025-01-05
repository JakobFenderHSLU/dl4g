import os
import subprocess


def get_command(play_rule_strategies: str):
    return (f"python run.py --seed 42 --n_games 200 "
            f"--agent-play-strategy dmcts --agent-trump-strategy deep_nn "
            f"--opponent-play-strategy dmcts --opponent-trump-strategy deep_nn "
            f"--agent-play-rule-strategies {play_rule_strategies}").split(" ")


play_rules = [
    "none",
    "only_valid",
    "smear",
    "mini_max",
    "trump_jack",
    "pull_trumps",
    "all"
]

current_env = os.environ.copy()
for play_rule in play_rules:
    command = get_command(play_rule)
    print(f"Running command: {command}")
    subprocess.run(command, env=current_env, shell=True)
    print(f"Command finished: {command}")
