using MLNet;
using MLNet.Losses;
using MLNet.Metrics;
using MLNet.Models.Regression;
using MLNet.Optimizers;
using MLNet.Utils;
using Numpy;
using Xunit;
using Xunit.Abstractions;

namespace MLNetTest.Regression
{
    public class PolyRegression : AbstractUnitTest
    {
        private readonly string singledata = @"..\..\..\..\DataSet\data_singlevar.txt";

        public PolyRegression(ITestOutputHelper testOutputHelper)
            : base(testOutputHelper)
        {
            var data = np2.load(singledata);

            X = data[":,0:1"];
            Y = data[":,1:2"];
        }

        protected NDarray X { set; get; }
        protected NDarray Y { set; get; }


        [Fact]
        public void PolynomialFeatures()
        {
            var pr = new MLNet.Models.Regression.PolyRegression();
            var trainPlan = new TrainPlan(100, learningRate: 1E-1);

            pr.GiveOptimizer(new Adam(trainPlan.LearningRate));
            pr.GiveLoss(new LSLoss {Constraint = Constraint.None});
            pr.GiveMetric(new MSE(), new MAE());
            pr.Fit(X, Y, trainPlan);

            print(pr);
        }
    }
}