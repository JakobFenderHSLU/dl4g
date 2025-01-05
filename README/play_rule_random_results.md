# Results for Play Rules

PlayRules Tested:

- `None`: No play rule.
- `OnlyValid`: If only one card is valid, play it.
- `Smear`: Smearing is a strategy in Jass where you play a high value card when the trick is already won.
- `TrumpJack`: If player has Trump Jack, play it if an opponent played the Trump 9 or the total points of the
  trick are above 20.
- `PullTrump`: As long as the opponents have trump cards, play the highest trump card.
- `MiniMax`: At a certain threshold, switch to mini-max strategy.

Agent:

- Trump Strategy: DeepNNTrumpStrategy
- Play Strategy: RandomPlayStrategy

Opponent:

- Trump Strategy: DeepNNTrumpStrategy
- Play Strategy: RandomPlayStrategy

Note:

- 5 seconds per move for MiniMaxPlayRule

Command used:

```bash
run.py --seed 42 --n_games 10000 --agent-play-strategy <strategy> --agent-trump-strategy deep_nn --opponent-trump-strategy deep_nn
```

## Results

### None

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

### OnlyValid

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

### Smear

<pre>
 5108                                          4892
[+++++++++++++++++++++++++------------------------]
                                                   
+------------------------+---------+-----------+   
|        Overall         |  Agents | Opponents |   
+------------------------+---------+-----------+   
|        Winrate         | 51.08 % |  48.92 %  |   
|     Average Points     | 79.3933 |  77.6067  |   
|       Max Points       |  157.0  |   157.0   |   
|       Min Points       |   0.0   |    0.0    |   
| Points 25th Percentile |   51.0  |    49.0   |   
|     Points Median      |   80.0  |    77.0   |   
| Points 75th Percentile |  108.0  |   106.0   |   
+------------------------+---------+-----------+   
                                                   
+------------------------+----------+-----------+ 
|      Trump Rounds      |  Agents  | Opponents | 
+------------------------+----------+-----------+ 
|        Winrate         | 81.50 %  |  79.34 %  | 
|     Average Points     | 103.7926 |  102.006  | 
|       Max Points       |  157.0   |   157.0   | 
|       Min Points       |   0.0    |    0.0    | 
| Points 25th Percentile |   85.0   |    83.0   | 
|     Points Median      |  105.0   |   104.0   | 
| Points 75th Percentile |  124.0   |   122.0   | 
+------------------------+----------+-----------+ 
</pre>

### TrumpJack

<pre>
 5057                                          4943
[+++++++++++++++++++++++++------------------------]
                                                   
+------------------------+---------+-----------+   
|        Overall         |  Agents | Opponents |   
+------------------------+---------+-----------+   
|        Winrate         | 50.57 % |  49.43 %  |   
|     Average Points     | 79.2585 |  77.7415  |   
|       Max Points       |  157.0  |   157.0   |   
|       Min Points       |   0.0   |    0.0    |   
| Points 25th Percentile |   51.0  |    49.0   |   
|     Points Median      |   79.0  |    78.0   |   
| Points 75th Percentile |  108.0  |   106.0   |   
+------------------------+---------+-----------+   
                                                   
+------------------------+----------+-----------+ 
|      Trump Rounds      |  Agents  | Opponents | 
+------------------------+----------+-----------+ 
|        Winrate         | 81.54 %  |  80.40 %  | 
|     Average Points     | 104.3382 |  102.8212 | 
|       Max Points       |  157.0   |   157.0   | 
|       Min Points       |   10.0   |    0.0    | 
| Points 25th Percentile |   85.0   |    84.0   | 
|     Points Median      |  106.0   |   103.0   | 
| Points 75th Percentile |  125.0   |   123.0   | 
+------------------------+----------+-----------+ 
</pre>

### PullTrump

<pre>
 4929                                          5071
[++++++++++++++++++++++++-------------------------]
                                                   
+------------------------+---------+-----------+   
|        Overall         |  Agents | Opponents |   
+------------------------+---------+-----------+   
|        Winrate         | 49.29 % |  50.71 %  |   
|     Average Points     | 79.0959 |  77.9041  |   
|       Max Points       |  157.0  |   157.0   |   
|       Min Points       |   0.0   |    0.0    |   
| Points 25th Percentile |  49.75  |    48.0   |   
|     Points Median      |   78.0  |    79.0   |   
| Points 75th Percentile |  109.0  |   107.25  |   
+------------------------+---------+-----------+   
                                                   
+------------------------+----------+-----------+ 
|      Trump Rounds      |  Agents  | Opponents | 
+------------------------+----------+-----------+ 
|        Winrate         | 80.18 %  |  81.60 %  | 
|     Average Points     | 104.5728 |  103.381  | 
|       Max Points       |  157.0   |   157.0   | 
|       Min Points       |   0.0    |    0.0    | 
| Points 25th Percentile |   84.0   |    85.0   | 
|     Points Median      |  106.0   |   105.0   | 
| Points 75th Percentile |  127.25  |   124.0   | 
+------------------------+----------+-----------+ 
</pre>

### MiniMax

<pre>
 51                                              49
[+++++++++++++++++++++++++------------------------]
                                                   
+------------------------+---------+-----------+   
|        Overall         |  Agents | Opponents |   
+------------------------+---------+-----------+   
|        Winrate         | 51.00 % |  49.00 %  |   
|     Average Points     |  78.46  |   78.54   |   
|       Max Points       |  157.0  |   157.0   |   
|       Min Points       |   0.0   |    0.0    |   
| Points 25th Percentile |   47.0  |    48.5   |   
|     Points Median      |   79.5  |    77.5   |   
| Points 75th Percentile |  108.5  |   110.0   |   
+------------------------+---------+-----------+   
                                                   
+------------------------+---------+-----------+  
|      Trump Rounds      |  Agents | Opponents |  
+------------------------+---------+-----------+  
|        Winrate         | 84.00 % |  82.00 %  |  
|     Average Points     |  105.7  |   105.78  |  
|       Max Points       |  157.0  |   157.0   |  
|       Min Points       |   18.0  |    38.0   |  
| Points 25th Percentile |   90.5  |    89.0   |  
|     Points Median      |  106.5  |   108.5   |  
| Points 75th Percentile |  129.0  |   125.5   |  
+------------------------+---------+-----------+  
</pre>
