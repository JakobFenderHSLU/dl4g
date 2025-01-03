# Agent for DL4G

This repository contains the code for the bots that are used in the Deep Learning for Games course at the Lucerne
University of Applied Sciences and Arts. It is based on the Jass game, which is a popular card game in Switzerland.
If you are not familiar with the game, you can find more information
[here](https://www.swisslos.ch/en/jass/informations/jass-rules/principles-of-jass.html).

## Authors

- [Jakob Fender](https://github.com/JakobFenderHSLU)
- [Dan Livingston](https://github.com/danlivingston)

## Run the bots

The run command will create an Arena and run the bots against each other. It will create four bots, two of each team.
After the half of the games have been played a new arena will be created with the bots switched places. Get more
information about how to run the arena by running the following command:

```bash
python run.py --help
```

## Strategies

In this simulation of Jass the bots have to implement two methods.

- `choose_card` which is called when the bot has to choose a card to play.
- `choose_trump` which is called when the bot has to choose the trump.

We have decided to create strategies for each method. This allows us to try different trump and play strategy
combinations.

### Trump Strategies

A trump strategy receives an Observation of a game, and it has to return a trump:

- `DIAMONDS`
- `HEARTS`
- `SPADES`
- `CLUBS`
- `OBE_ABE`
- `UNE_UFE`

Or it has the option to choose `PUSH`. This lets the partner choose the trump.

### Implemented Trump Strategies

- `RandomTrumpStrategy` chooses a random trump. This strategy can't choose `PUSH`.
- `HighestSumTrumpStrategy` chooses the trump with the most cards of the same color. This strategy won't choose
  OBE_ABE or UNE_UFE and is unable to `PUSH`.
- `HighestScoreTrumpStrategy` calculates a score for each trump and chooses one if it is above a certain threshold.
  This strategy was proposed by Daniel Graf in
  his [matura work](https://dgraf.ch/d/Kanti/Jassen_auf_Basis_der_Spieltheorie-Daniel_Graf.pdf).
- `StatisticalTrumpStrategy` uses a statistical approach to choose the trump. It is based on a dataset of 1.8 Mio games
  played on [swisslos.ch](https://www.swisslos.ch/en/jass/schieber/play.html). It calculates how often a card was in the
  hand of the player when he picked a trump.
- `DeepNNTrumpStrategy` was trained on synthetic data. We generated **~2'000'000** hands and played **20** games for
  every trump with a random play strategy. This means we played **120'000'000** games in total. Then we trained a Simple
  NN to predict the average amount of points the player with that hand would make. For the Trump selection we chose the
  highest score predicted. If the predicted hand score would be below a certain threshold we would PUSH instead.

### Play Strategies

This strategy is way more complex than the trump strategy. It receives an observation of the game containing the
following information:

- the dealer
- the player that declared trump,
- the trump chosen
- trump was declared forehand (derived information)
- the tricks that have been played so far
- the winner and the first player (derived) of each trick
- the number of points that have been made by the current jass_players team in this round
- the number of points that have been made by the opponent team in this round
- the number of cards played in the current trick
- the cards played in the current trick
- the current player
- the hand of the player

From this information, the bot has to decide which card on his hand to play. This is a complex problem and can be
solved in many different ways.

### Implemented Play Strategies

- `RandomPlayStrategy` chooses a random valid card from the hand.
- `HighestValuePlayStrategy` chooses the highest value card from the hand.
- `MCTSPlayStrategy` Randomly distributes other cards among the players. Then uses Monte Carlo Tree Search to find the
  best card to play for this hand.
- `DeterminizedMCTSPlayStrategy` Takes **n** random samples of the remaining cards and uses Monte Carlo Tree Search to
  find the best card to play for this hand. Takes the card that works best on average.

### Play Rules

This is a list of rules that will trigger before a PlayStrategy is called. This is in order to play no-brainer moves.
The Strategy will go through the list and if one is triggered, it will play the card and return it. If no rule is
triggered, the PlayStrategy will be called.

- `OnlyValidPlayRule` if only one card is valid, play it. This is to save time and resources.
- `SmearPlayRule` Smearing is a strategy in Jass where you play a high value card when the trick is already won.
- `TrumpJackPlayRule` If player has Trump Jack, play it if an opponent played the Trump 9 or the total points of the
  trick are above 20.
- `PullTrumpPlayRule` As long as the opponents have trump cards, play the highest trump card.
- `MiniMaxPlayRule` At a certain threshold, switch to mini-max strategy.
- `SwisslosOpeningPlayRule` This was an attempt to code all opening rules from swisslos. We discontinued this because
  it was too complex and not worth the effort.

## Generate Data

We have also implemented a script that generates the average score per trump that a player for a random hand. This can
be used to train a model that predicts the points that a player will make with all trumps. For more information run the
following command:

```bash
python generate_trump_data.py --help
```

## Evaluation

### Trump Strategies

To evaluate the trump strategies independent of the play strategy, we ran every Trump strategy together with the
`random` play strategy and no play rules. The opponent played with a random trump strategy. We played 10'000 games in
total. After 5'000 the Arena was reset and the positions were swapped. This ensures, that the `random` play strategies
returned the same outputs after every half.

Command used:

```bash
run.py --seed 42 --n_games 10000 --agent-trump-strategy <strategy> 
```

O = overall, T = in Trump Rounds

|              | Winrate O | Winrate T | Average Points O | Average Points T |
|--------------|-----------|-----------|------------------|------------------|
| Random       | 50.00 %   | 50.00 %   | 78.5             | 78.5             |
| HighestSum   | 59.15 %   | 69.88 %   | 85.8071          | 93.7382          |
| HighestScore | 64.11 %   | 78.44 %   | 89.8663          | 101.387          |
| Statistical  | 62.43 %   | 75.24 %   | 88.2777          | 98.3564          |
| DeepNN       | 65.23 %   | 81.08 %   | 90.6736          | 103.2246         |

For a more detailed evaluation see [evaluation_results.md](README/evaluation_results.md).

### Play Strategies

Similar to the Trump strategies, we evaluated the play strategies independent of the trump strategy. We ran every Play
strategy together with the best Trump strategy `DeepNNTrumpStrategy` for both teams. The amount of games we played
varied due to computational limitations. The timelimit for choosing a card is 5 seconds, half of the time in the
official tournament.

Command used:

```bash
run.py --seed 42 --n_games 10000 --agent-play-strategy <strategy> --agent-trump-strategy deep_nn --opponent-trump-strategy deep_nn
```

O = overall, T = in Trump Rounds

|                   | Winrate O | Winrate T | Average Points O | Average Points T | Games Played |
|-------------------|-----------|-----------|------------------|------------------|--------------|
| Random            | 50.00 %   | 80.74 %   | 78.5             | 102.9398         | 10'000       |
| HighestValue      | 46.14 %   | 79.40 %   | 75.6356          | 104.0712         | 10'000       |
| MCTS              | 62.00 %   | 98.00 %   | 88.9             | 117.96           | 100          |
| Determinized MCTS | 64.00 %   | 96.00 %   | 90.61            | 117.1            | 100          |

### Play Rule Strategies

For the Play Rule Strategies, we evaluated them together with DeepNNTrumpStrategy. For the Play strategie, we selected
RandomPlayStrategy and DeterminizedMCTSPlayStrategy. The amount of games we played varied due to computational
limitations. Note that MiniMaxPlayRule was limited to 5 seconds per move.

Command used:

```bash
run.py --seed 42 --n_games 10000 --agent-play-strategy random --agent-trump-strategy deep_nn --opponent-trump-strategy deep_nn --agent-play-rule-strategies <strategy>
run.py --seed 42 --n_games 100 --agent-play-strategy dmcts --opponent-play-strategy dmcts --agent-trump-strategy deep_nn --opponent-trump-strategy deep_nn --agent-play-rule-strategies <strategy>
```

#### RandomPlayStrategy

O = overall, T = in Trump Rounds

|           | Winrate O | Winrate T | Average Points O | Average Points T | Games Played |
|-----------|-----------|-----------|------------------|------------------|--------------|
| None      | 50.00 %   | 80.74 %   | 78.5             | 102.9398         | 10'000       |
| OnlyValid | 50.00 %   | 80.74 %   | 78.5             | 102.9398         | 10'000       |
| Smear     | 51.08 %   | 81.50 %   | 79.3933          | 103.7926         | 10'000       |
| TrumpJack | 50.57 %   | 81.54 %   | 79.2585          | 104.3382         | 10'000       |
| PullTrump | 49.29 %   | 80.18 %   | 79.0959          | 104.5728         | 10'000       |
| MiniMax   | 51.00 %   | 84.00 %   | 78.46            | 105.7            | 100          |

#### DeterminizedMCTSPlayStrategy

O = overall, T = in Trump Rounds

|           | Winrate O | Winrate T | Average Points O | Average Points T | Games Played |
|-----------|-----------|-----------|------------------|------------------|--------------|
| None      | 46.00 %   | 60.00 %   | 79.22            | 95.64            | 50           |
| OnlyValid | 56.00 %   | 88.00 %   | 81.24            | 99.6             | 50           |
| Smear     | 54.00 %   | 80.00 %   | 80.9             | 102.8            | 50           |
| TrumpJack | 48.00 %   | 76.00 %   | 77.44            | 99.52            | 50           |