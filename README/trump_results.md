# Results for TrumpStrategy

TrumpStrategies Tested:

- `Random` chooses a random trump. This strategy can't choose `PUSH`.
- `HighestSum` chooses the trump with the most cards of the same color. This strategy won't choose
  OBE_ABE or UNE_UFE and is unable to `PUSH`.
- `HighestScore` calculates a score for each trump and chooses one if it is above a certain threshold.
  This strategy was proposed by Daniel Graf in
  his [matura work](https://dgraf.ch/d/Kanti/Jassen_auf_Basis_der_Spieltheorie-Daniel_Graf.pdf).
- `Statistical` uses a statistical approach to choose the trump. It is based on a dataset of 1.8 Mio games
  played on [swisslos.ch](https://www.swisslos.ch/en/jass/schieber/play.html). It calculates how often a card was in the
  hand of the player when he picked a trump.
- `DeepNN` was trained on synthetic data. We generated **~2'000'000** hands and played **20** games for
  every trump with a random play strategy. This means we played **120'000'000** games in total. Then we trained a Simple
  NN to predict the average amount of points the player with that hand would make. For the Trump selection we chose the
  highest score predicted. If the predicted hand score would be below a certain threshold we would PUSH instead.

Agent:

- Play Strategy: RandomPlayStrategy
- Play Rules: None

Opponent:

- Trump Strategy: DeepNNTrumpStrategy
- Play Rules: None

Command used:

```bash
run.py --seed 42 --n_games 10000 --agent-trump-strategy <strategy> 
```

## Results

#### Random

<pre>                         
 5000                                          5000 
[+++++++++++++++++++++++++-------------------------]
                                                    
+------------------------+---------+-----------+   
|        Overall         |  Agents | Opponents |   
+------------------------+---------+-----------+   
|        Winrate         | 50.00 % |  50.00 %  |   
|     Average Points     |   78.5  |    78.5   |   
|       Max Points       |  157.0  |   157.0   |   
|       Min Points       |   0.0   |    0.0    |   
| Points 25th Percentile |   55.0  |    55.0   |   
|     Points Median      |   78.5  |    78.5   |   
| Points 75th Percentile |  102.0  |   102.0   |   
+------------------------+---------+-----------+   
                                                   
+------------------------+---------+-----------+   
|      Trump Rounds      |  Agents | Opponents |   
+------------------------+---------+-----------+   
|        Winrate         | 50.06 % |  50.06 %  |   
|     Average Points     | 78.5858 |  78.5858  |   
|       Max Points       |  157.0  |   157.0   |   
|       Min Points       |   0.0   |    0.0    |   
| Points 25th Percentile |   55.0  |    55.0   |   
|     Points Median      |   79.0  |    79.0   |   
| Points 75th Percentile |  102.0  |   102.0   |   
+------------------------+---------+-----------+     
</pre>

#### Highest Sum

<pre>
 5915                                          4085
[+++++++++++++++++++++++++++++--------------------]
                                                   
+------------------------+---------+-----------+  
|        Overall         |  Agents | Opponents |  
+------------------------+---------+-----------+  
|        Winrate         | 59.15 % |  40.85 %  |  
|     Average Points     | 85.8071 |  71.1929  |  
|       Max Points       |  157.0  |   157.0   |  
|       Min Points       |   0.0   |    0.0    |  
| Points 25th Percentile |   63.0  |    47.0   |  
|     Points Median      |   87.0  |    70.0   |  
| Points 75th Percentile |  110.0  |    94.0   |  
+------------------------+---------+-----------+  
                                                  
+------------------------+---------+-----------+  
|      Trump Rounds      |  Agents | Opponents |  
+------------------------+---------+-----------+  
|        Winrate         | 69.88 % |  51.58 %  |  
|     Average Points     | 93.7382 |   79.124  |  
|       Max Points       |  157.0  |   157.0   |  
|       Min Points       |   0.0   |    0.0    |  
| Points 25th Percentile |   73.0  |    55.0   |  
|     Points Median      |   95.0  |    80.0   |  
| Points 75th Percentile |  115.25 |   103.0   |  
+------------------------+---------+-----------+  
</pre>

#### Highest Score

<pre>
Total Games Played: 10000                          
 6411                                          3589
[++++++++++++++++++++++++++++++++-----------------]
                                                   
+------------------------+---------+-----------+  
|        Overall         |  Agents | Opponents |  
+------------------------+---------+-----------+  
|        Winrate         | 64.11 % |  35.89 %  |  
|     Average Points     | 89.8663 |  67.1337  |  
|       Max Points       |  157.0  |   157.0   |  
|       Min Points       |   0.0   |    0.0    |  
| Points 25th Percentile |   66.0  |    42.0   |  
|     Points Median      |   92.0  |    65.0   |  
| Points 75th Percentile |  115.0  |    91.0   |  
+------------------------+---------+-----------+  
                                                  
+------------------------+---------+-----------+  
|      Trump Rounds      |  Agents | Opponents |  
+------------------------+---------+-----------+  
|        Winrate         | 78.44 % |  50.22 %  |  
|     Average Points     | 101.387 |  78.6544  |  
|       Max Points       |  157.0  |   157.0   |  
|       Min Points       |   6.0   |    0.0    |  
| Points 25th Percentile |   82.0  |    55.0   |  
|     Points Median      |  103.0  |    79.0   |  
| Points 75th Percentile |  123.0  |   102.0   |  
+------------------------+---------+-----------+  
</pre>

#### Statistical

<pre>
 6243                                          3757
[+++++++++++++++++++++++++++++++------------------]
                                                   
+------------------------+---------+-----------+  
|        Overall         |  Agents | Opponents |  
+------------------------+---------+-----------+  
|        Winrate         | 62.43 % |  37.57 %  |  
|     Average Points     | 88.2777 |  68.7223  |  
|       Max Points       |  157.0  |   157.0   |  
|       Min Points       |   0.0   |    0.0    |  
| Points 25th Percentile |   66.0  |    45.0   |  
|     Points Median      |   89.0  |    68.0   |  
| Points 75th Percentile |  112.0  |    91.0   |  
+------------------------+---------+-----------+  
                                                  
+------------------------+---------+-----------+  
|      Trump Rounds      |  Agents | Opponents |  
+------------------------+---------+-----------+  
|        Winrate         | 75.24 % |  50.38 %  |  
|     Average Points     | 98.3564 |   78.801  |  
|       Max Points       |  157.0  |   157.0   |  
|       Min Points       |   0.0   |    0.0    |  
| Points 25th Percentile |   79.0  |    55.0   |  
|     Points Median      |   99.0  |    79.0   |  
| Points 75th Percentile |  118.0  |   103.0   |  
+------------------------+---------+-----------+  
</pre>

#### Deep NN

<pre>
 6523                                          3477
[++++++++++++++++++++++++++++++++-----------------]
                                                   
+------------------------+---------+-----------+  
|        Overall         |  Agents | Opponents |  
+------------------------+---------+-----------+  
|        Winrate         | 65.23 % |  34.77 %  |  
|     Average Points     | 90.6736 |  66.3264  |  
|       Max Points       |  157.0  |   157.0   |  
|       Min Points       |   0.0   |    0.0    |  
| Points 25th Percentile |   68.0  |    42.0   |  
|     Points Median      |   93.0  |    64.0   |  
| Points 75th Percentile |  115.0  |    89.0   |  
+------------------------+---------+-----------+  
                                                  
+------------------------+----------+-----------+ 
|      Trump Rounds      |  Agents  | Opponents | 
+------------------------+----------+-----------+ 
|        Winrate         | 81.08 %  |  50.62 %  | 
|     Average Points     | 103.2246 |  78.8774  | 
|       Max Points       |  157.0   |   157.0   | 
|       Min Points       |   0.0    |    0.0    | 
| Points 25th Percentile |   85.0   |    55.0   | 
|     Points Median      |  105.0   |    79.0   | 
| Points 75th Percentile |  124.0   |   102.0   | 
+------------------------+----------+-----------+ 
</pre>
