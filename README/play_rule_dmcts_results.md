# Results for Play Rules

PlayRules Tested:

- `None`: No play rule.
- `OnlyValid`: If only one card is valid, play it.
- `Smear`: Smearing is a strategy in Jass where you play a high value card when the trick is already won.
- `TrumpJack`: If player has Trump Jack, play it if an opponent played the Trump 9 or the total points of the
  trick are above 20.
- `PullTrump`: As long as the opponents have trump cards, play the highest trump card.
- `MiniMax`: At a certain threshold, switch to mini-max strategy.
- `All`: All Rules combined.

Agent:

- Trump Strategy: DeepNNTrumpStrategy
- Play Strategy: DeterminizedMCTSPlayStrategy

Opponent:

- Trump Strategy: DeepNNTrumpStrategy
- Play Strategy: DeterminizedMCTSPlayStrategy

Note:

- 5 seconds per move for MiniMaxPlayRule
- 5 seconds per move for DeterminizedMCTSPlayRule

#### Command

```bash
run.py --seed 42 --n_games 200 --agent-play-strategy dmcts --opponent-play-strategy dmcts --agent-trump-strategy deep_nn --opponent-trump-strategy deep_nn --agent-play-rule-strategies <strategy>
```

## Results

### None

<pre>
 97                                             103
[++++++++++++++++++++++++-------------------------]
                                                   
+------------------------+---------+-----------+   
|        Overall         |  Agents | Opponents |   
+------------------------+---------+-----------+   
|        Winrate         | 48.50 % |  51.50 %  |   
|     Average Points     |  77.85  |   79.15   |   
|       Max Points       |  157.0  |   157.0   |   
|       Min Points       |   0.0   |    0.0    |   
| Points 25th Percentile |  51.75  |   51.75   |   
|     Points Median      |   77.0  |    80.0   |   
| Points 75th Percentile |  105.25 |   105.25  |   
+------------------------+---------+-----------+   
                                                   
+------------------------+---------+-----------+  
|      Trump Rounds      |  Agents | Opponents |  
+------------------------+---------+-----------+  
|        Winrate         | 81.00 % |  84.00 %  |  
|     Average Points     |  103.62 |   104.92  |  
|       Max Points       |  157.0  |   157.0   |  
|       Min Points       |   35.0  |    51.0   |  
| Points 25th Percentile |   85.5  |    85.0   |  
|     Points Median      |  105.0  |   104.5   |  
| Points 75th Percentile |  122.25 |   118.5   |  
+------------------------+---------+-----------+
</pre>

### OnlyValid

<pre>
 105                                             95
[++++++++++++++++++++++++++-----------------------]
                                                   
+------------------------+---------+-----------+   
|        Overall         |  Agents | Opponents |   
+------------------------+---------+-----------+   
|        Winrate         | 52.50 % |  47.50 %  |   
|     Average Points     |  78.745 |   78.255  |   
|       Max Points       |  157.0  |   157.0   |   
|       Min Points       |   0.0   |    0.0    |   
| Points 25th Percentile |  52.75  |    53.0   |   
|     Points Median      |   82.0  |    75.0   |   
| Points 75th Percentile |  104.0  |   104.25  |   
+------------------------+---------+-----------+   
                                                   
+------------------------+---------+-----------+  
|      Trump Rounds      |  Agents | Opponents |  
+------------------------+---------+-----------+  
|        Winrate         | 88.00 % |  83.00 %  |  
|     Average Points     |  106.4  |   105.91  |  
|       Max Points       |  157.0  |   157.0   |  
|       Min Points       |   60.0  |    46.0   |  
| Points 25th Percentile |  88.75  |   85.75   |  
|     Points Median      |  103.5  |   104.5   |  
| Points 75th Percentile |  126.5  |   126.25  |  
+------------------------+---------+-----------+  
</pre>

### Smear

<pre>
 101                                             99
[+++++++++++++++++++++++++------------------------]
                                                   
+------------------------+---------+-----------+   
|        Overall         |  Agents | Opponents |   
+------------------------+---------+-----------+   
|        Winrate         | 50.50 % |  49.50 %  |   
|     Average Points     |  78.575 |   78.425  |   
|       Max Points       |  157.0  |   157.0   |   
|       Min Points       |   0.0   |    0.0    |   
| Points 25th Percentile |  55.75  |   54.75   |   
|     Points Median      |   79.0  |    78.0   |   
| Points 75th Percentile |  102.25 |   101.25  |   
+------------------------+---------+-----------+   
                                                   
+------------------------+---------+-----------+  
|      Trump Rounds      |  Agents | Opponents |  
+------------------------+---------+-----------+  
|        Winrate         | 84.00 % |  83.00 %  |  
|     Average Points     |  103.07 |   102.92  |  
|       Max Points       |  157.0  |   157.0   |  
|       Min Points       |   23.0  |    54.0   |  
| Points 25th Percentile |   85.0  |    86.0   |  
|     Points Median      |  102.0  |   100.0   |  
| Points 75th Percentile |  120.0  |   120.25  |  
+------------------------+---------+-----------+  
</pre>

### MiniMax

<pre>
 95                                             105
[+++++++++++++++++++++++--------------------------]
                                                   
+------------------------+---------+-----------+   
|        Overall         |  Agents | Opponents |   
+------------------------+---------+-----------+   
|        Winrate         | 47.50 % |  52.50 %  |   
|     Average Points     |  74.82  |   82.18   |   
|       Max Points       |  157.0  |   157.0   |   
|       Min Points       |   0.0   |    0.0    |   
| Points 25th Percentile |  48.75  |   58.75   |   
|     Points Median      |   76.5  |    80.5   |   
| Points 75th Percentile |  98.25  |   108.25  |   
+------------------------+---------+-----------+   
                                                   
+------------------------+---------+-----------+  
|      Trump Rounds      |  Agents | Opponents |  
+------------------------+---------+-----------+  
|        Winrate         | 79.00 % |  84.00 %  |  
|     Average Points     |  96.63  |   103.99  |  
|       Max Points       |  157.0  |   157.0   |  
|       Min Points       |   18.0  |    29.0   |  
| Points 25th Percentile |   83.0  |   85.75   |  
|     Points Median      |   97.5  |   105.5   |  
| Points 75th Percentile |  112.25 |   123.0   |  
+------------------------+---------+-----------+  
</pre>

### Trump Jack

<pre>
 105                                             95
[++++++++++++++++++++++++++-----------------------]
                                                    
+------------------------+---------+-----------+   
|        Overall         |  Agents | Opponents |   
+------------------------+---------+-----------+   
|        Winrate         | 52.50 % |  47.50 %  |   
|     Average Points     |  77.88  |   79.12   |   
|       Max Points       |  157.0  |   157.0   |   
|       Min Points       |   0.0   |    0.0    |   
| Points 25th Percentile |   54.0  |    51.0   |   
|     Points Median      |   80.0  |    77.0   |   
| Points 75th Percentile |  106.0  |   103.0   |   
+------------------------+---------+-----------+   
                                                    
+------------------------+---------+-----------+  
|      Trump Rounds      |  Agents | Opponents |  
+------------------------+---------+-----------+  
|        Winrate         | 83.00 % |  78.00 %  |  
|     Average Points     |  103.16 |   104.4   |  
|       Max Points       |  157.0  |   157.0   |  
|       Min Points       |   42.0  |    23.0   |  
| Points 25th Percentile |   86.0  |   82.75   |  
|     Points Median      |  105.5  |   101.0   |  
| Points 75th Percentile |  121.0  |   126.75  |  
+------------------------+---------+-----------+  
</pre>

### Pull Trumps

<pre>
 95                                             105
[+++++++++++++++++++++++-------------------------
                                                 
+------------------------+---------+-----------+ 
|        Overall         |  Agents | Opponents | 
+------------------------+---------+-----------+ 
|        Winrate         | 47.50 % |  52.50 %  | 
|     Average Points     |  77.485 |   79.515  | 
|       Max Points       |  157.0  |   157.0   | 
|       Min Points       |   0.0   |    0.0    | 
| Points 25th Percentile |   54.0  |   53.25   | 
|     Points Median      |   76.5  |    80.5   | 
| Points 75th Percentile |  103.75 |   103.0   | 
+------------------------+---------+-----------+ 
                                                 
+------------------------+---------+-----------+
|      Trump Rounds      |  Agents | Opponents |
+------------------------+---------+-----------+
|        Winrate         | 74.00 % |  79.00 %  |
|     Average Points     |  100.78 |   102.81  |
|       Max Points       |  157.0  |   157.0   |
|       Min Points       |   43.0  |    41.0   |
| Points 25th Percentile |   78.0  |   82.75   |
|     Points Median      |   98.5  |   100.0   |
| Points 75th Percentile |  119.75 |   120.75  |
+------------------------+---------+-----------+   
</pre>

### All Play Rules

<pre>
 102                                             98
[+++++++++++++++++++++++++------------------------]
                                                   
 +------------------------+---------+-----------+  
 |        Overall         |  Agents | Opponents |  
 +------------------------+---------+-----------+  
 |        Winrate         | 51.00 % |  49.00 %  |  
 |     Average Points     |  78.79  |   78.21   |  
 |       Max Points       |  157.0  |   157.0   |  
 |       Min Points       |   0.0   |    0.0    |  
 | Points 25th Percentile |   54.0  |    52.0   |  
 |     Points Median      |   80.0  |    77.0   |  
 | Points 75th Percentile |  105.0  |   103.0   |  
 +------------------------+---------+-----------+  
                                                   
 +------------------------+---------+-----------+ 
 |      Trump Rounds      |  Agents | Opponents | 
 +------------------------+---------+-----------+ 
 |        Winrate         | 83.00 % |  81.00 %  | 
 |     Average Points     |  103.18 |   102.6   | 
 |       Max Points       |  157.0  |   157.0   | 
 |       Min Points       |   21.0  |    52.0   | 
 | Points 25th Percentile |   86.0  |    83.0   | 
 |     Points Median      |  105.0  |   103.0   | 
 | Points 75th Percentile |  120.25 |   120.25  | 
 +------------------------+---------+-----------+ 
</pre>