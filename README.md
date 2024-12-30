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

#### Overview

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
| Random       | 50.00 %   | 50.06 %   | 78.5             | 78.5858          |
| HighestSum   | 59.15 %   | 69.88 %   | 85.8071          | 93.7382          |
| HighestScore | 64.11 %   | 78.44 %   | 89.8663          | 101.387          |
| Statistical  | 62.43 %   | 75.24 %   | 88.2777          | 98.3564          |
| DeepNN       | 65.23 %   | 81.08 %   | 90.6736          | 103.2246         |

For a more detailed evaluation see [evaluation_results.md](README/evaluation_results.md).
