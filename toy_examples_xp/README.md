# Toy examples

## Case 1 and 2

To get the data execute ```simulateR_2.py```. By default, it will launch an experiment for Case 1 with $M=5000$ runs of AdaStop algorithm. To get the data for other experiments one should manually modify those parameters in the file:
```python 
  M=5000
  EXP = "exp1"
```
Note that ```EXP``` can be ```"exp1"``` for Case 1 and ```"exp2"``` for Case 2. As a result the code will generate two .csv files. One for decisions and one for the effective number of seeds that AdaStop performs. For example, if one executes the file for Case 1 with $M=5000$ then the files generated will be ```exp1_5000_decs.csv``` and ```exp1_5000_niter.csv```

To reproduce the plot for Cases 1 and 2 from paper, run ```plot_cases12.py```. By default, it will built the plots using files ```exp1_5000_decs.csv```, ```exp1_5000_niter.csv```, ```exp2_5000_niter.csv``` and ```exp2_5000_niter.csv```. If you want to use the other data you should change those lines:
```python
    powers, power_stds, power_confidence_intervals = powers_case("exp2_5000_decs.csv", "exp2_5000_niter.csv")
    powers2, power_stds2, power_confidence_intervals2 = powers_case("exp1_5000_decs.csv", "exp1_5000_niter.csv")
```


## Case 3

To estimate Family-Wise-Error, one should execute ```simulatedR_multi.py```. One can play with parameter $M$, the one used for the paper is $M=5000$

To get the plot with decisions and empirically measured distributions, run ```simulatedR_multi_1seed.py```
