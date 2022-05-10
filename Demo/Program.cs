// See https://aka.ms/new-console-template for more information

using MLNet;
using MLNet.Losses;
using MLNet.Metrics;
using MLNet.Models.Regression;
using MLNet.Optimizers;
using Numpy;

Log.print = Console.WriteLine;
Console.WriteLine("Start");
var x = Enumerable.Range(0, 200).Select(a => a * 0.01 - 2).ToArray();
var X = np.expand_dims(np.array(x), -1);
var Y = -2 + 1.5 * X - 3 * np.square(X);


Log.print(X);
Log.print(Y);

var pr = new PolyRegression(2);
var trainPlan = new TrainPlan(1000, learningRate: 1E-1);

pr.GiveOptimizer(new Adam(trainPlan.LearningRate));
pr.GiveLoss(new LSLoss {Constraint = Constraint.None});
pr.GiveMetric(new MSE(), new MAE());
pr.Fit(X, Y, trainPlan);