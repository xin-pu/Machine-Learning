using MLNet;
using MLNet.Losses;
using MLNet.Metrics;
using MLNet.Models;
using MLNet.Optimizers;
using MLNet.Utils;

namespace Demo
{
    public class PerceptronTask
    {
        private static readonly string singledata = @"..\..\..\..\DataSet\iris.data";

        public static void Run()
        {
            var (X, Y, Dict) = np2.loadClassifyData(singledata);
            var classes = Dict.Select(a => a.Key).ToArray();

            var perceptronper = new Perceptron(classes);
            var trainPlan = new TrainPlan(100, learningRate: 1E-1);
            perceptronper.GiveOptimizer(new Momentum(trainPlan.LearningRate));
            perceptronper.GiveLoss(new SoftmaxMutlitClassLoss(classes.Length));
            perceptronper.GiveMetric(new Accuracy(), new MacroPercision());
            perceptronper.Fit(X, Y, trainPlan);
            var y_pred = perceptronper.Predict(X);
            Log.print?.Invoke(y_pred);
            Log.print?.Invoke(perceptronper.Resolve);
        }
    }
}