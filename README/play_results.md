# Results for PlayStrategies

PlayStrategies Tested:

- `Random` chooses a random valid card from the hand.
- `HighestValue` chooses the highest value card from the hand.
- `MCTS` Randomly distributes other cards among the players. Then uses Monte Carlo Tree Search to find the
  best card to play for this hand.
- `DeterminizedMCTS` Takes **d** random samples of the remaining cards and uses Monte Carlo Tree Search to
  find the best card to play for this hand. Takes the card that works best on average.

Agent:

- Trump Strategy: DeepNNTrumpStrategy
- Play Rules: None

Opponent:

- Trump Strategy: DeepNNTrumpStrategy
- Play Rules: None

Note:

- 5 seconds per move and only 100 games for `MCTS` & `DeterminizedMCTS`

Command used:

```bash
run.py --seed 42 --n_games 10000 --agent-play-strategy dmcts --opponent-play-strategy dmcts --agent-trump-strategy deep_nn --opponent-trump-strategy deep_nn --agent-play-rule-strategies <strategy>
```

## Results

### Random

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
| Points 25th Percentile |   50.0  |    50.0   |
|     Points Median      |   78.5  |    78.5   |
| Points 75th Percentile |  107.0  |   107.0   |
+------------------------+---------+-----------+

+------------------------+----------+-----------+
|      Trump Rounds      |  Agents  | Opponents |
+------------------------+----------+-----------+
|        Winrate         | 80.74 %  |  80.74 %  |
|     Average Points     | 102.9398 |  102.9398 |
|       Max Points       |  157.0   |   157.0   |
|       Min Points       |   0.0    |    0.0    |
| Points 25th Percentile |   84.0   |    84.0   |
|     Points Median      |  104.0   |   104.0   |
| Points 75th Percentile |  123.0   |   123.0   |
+------------------------+----------+-----------+
</pre>

### HighestValue

<pre>
 4614                                          5386
[+++++++++++++++++++++++--------------------------]
                                                   
+------------------------+---------+-----------+   
|        Overall         |  Agents | Opponents |   
+------------------------+---------+-----------+   
|        Winrate         | 46.14 % |  53.86 %  |   
|     Average Points     | 75.6356 |  81.3644  |   
|       Max Points       |  157.0  |   157.0   |   
|       Min Points       |   0.0   |    0.0    |   
| Points 25th Percentile |   44.0  |    50.0   |   
|     Points Median      |   73.5  |    83.5   |   
| Points 75th Percentile |  107.0  |   113.0   |   
+------------------------+---------+-----------+   
                                                   
+------------------------+----------+-----------+ 
|      Trump Rounds      |  Agents  | Opponents | 
+------------------------+----------+-----------+ 
|        Winrate         | 79.40 %  |  87.12 %  | 
|     Average Points     | 104.0712 |   109.8   | 
|       Max Points       |  157.0   |   157.0   | 
|       Min Points       |   8.0    |    0.0    | 
| Points 25th Percentile |   83.0   |    93.0   | 
|     Points Median      |  105.0   |   112.0   | 
| Points 75th Percentile |  126.0   |   129.0   | 
+------------------------+----------+-----------+ 
</pre>

### MCTS

<pre>
 62                                              3
[+++++++++++++++++++++++++++++++------------------
                                                  
+------------------------+---------+-----------+  
|        Overall         |  Agents | Opponents |  
+------------------------+---------+-----------+  
|        Winrate         | 62.00 % |  38.00 %  |  
|     Average Points     |   88.9  |    68.1   |  
|       Max Points       |  157.0  |   152.0   |  
|       Min Points       |   5.0   |    0.0    |  
| Points 25th Percentile |  60.75  |    40.0   |  
|     Points Median      |   91.5  |    65.5   |  
| Points 75th Percentile |  117.0  |   96.25   |  
+------------------------+---------+-----------+  
                                                  
+------------------------+---------+-----------+ 
|      Trump Rounds      |  Agents | Opponents | 
+------------------------+---------+-----------+ 
|        Winrate         | 98.00 % |  74.00 %  | 
|     Average Points     |  117.96 |   97.16   | 
|       Max Points       |  157.0  |   152.0   | 
|       Min Points       |   71.0  |    40.0   | 
| Points 25th Percentile |  104.25 |   77.75   | 
|     Points Median      |  116.5  |    96.5   | 
| Points 75th Percentile |  135.5  |   118.75  | 
+------------------------+---------+-----------+ 
</pre>

### DeterminizedMCTS

<pre>
 64                                              36 
[++++++++++++++++++++++++++++++++------------------]
                                                    
+------------------------+---------+-----------+    
|        Overall         |  Agents | Opponents |    
+------------------------+---------+-----------+    
|        Winrate         | 64.00 % |  36.00 %  |    
|     Average Points     |  90.61  |   66.39   |    
|       Max Points       |  157.0  |   147.0   |    
|       Min Points       |   10.0  |    0.0    |    
| Points 25th Percentile |  62.75  |    38.0   |    
|     Points Median      |   97.5  |    59.5   |    
| Points 75th Percentile |  119.0  |   94.25   |    
+------------------------+---------+-----------+    
                                                    
+------------------------+---------+-----------+   
|      Trump Rounds      |  Agents | Opponents |   
+------------------------+---------+-----------+   
|        Winrate         | 96.00 % |  68.00 %  |   
|     Average Points     |  117.1  |   92.88   |   
|       Max Points       |  157.0  |   147.0   |   
|       Min Points       |   51.0  |    32.0   |   
| Points 25th Percentile |  102.0  |    73.5   |   
|     Points Median      |  118.0  |    93.0   |   
| Points 75th Percentile |  136.25 |   116.0   |   
+------------------------+---------+-----------+   
</pre>