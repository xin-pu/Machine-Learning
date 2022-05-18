using MLNet;
using MLNet.Losses;
using MLNet.Metrics;
using MLNet.Models;
using MLNet.Optimizers;
using Numpy;

namespace Demo
{
    public class LogicClassifyTask
    {
        public static void Run()
        {
            var x = Enumerable.Range(0, 200).Select(a => a * 0.01 - 2).ToArray();
            var X = np.expand_dims(np.array(x), -1);
            var Y = -2 + 1.5 * X - 3 * np.square(X);


            var pr = new Perceptron(Enumerable.Range(0, 4).ToArray());
            var trainPlan = new TrainPlan(1000, learningRate: 5E-2);

            pr.GiveOptimizer(new Momentum(trainPlan.LearningRate));
            pr.GiveLoss(new SoftmaxMutlitClassLoss(4));
            pr.GiveMetric(new Accuracy());
            pr.Fit(X, Y, trainPlan);
        }
    }
}