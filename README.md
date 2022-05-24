# Machine-Learning


## Demo
```
var pr = new MLNet.Models.Regression.PolyRegression();
var trainPlan = new TrainPlan(100, learningRate: 1E-1);

pr.GiveOptimizer(new InverseTime(trainPlan.LearningRate));
pr.GiveLoss(new LSLoss {Constraint = Constraint.Ridge});
pr.GiveMetric(new MSE(), new MAE());
pr.Fit(X, Y, trainPlan);
```

![File](Document/demolog.png)


## Frame

![frame](Document/总框架.png)

![frame](Document/优化算法.png)

![frame](Document/监督学习.png)

![frame](Document/无监督学习.png)
