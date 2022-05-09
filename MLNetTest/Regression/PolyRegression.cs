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
            //var x = Enumerable.Range(0, 200).Select(a => a * 0.05 - 2).ToArray();
            //X = np.expand_dims(np.array(x), -1);
            //Y = -2 + 1.5 * X + 2 * np.power(X, np.array(2));
        }

        protected NDarray X { set; get; }
        protected NDarray Y { set; get; }


        [Fact]
        public void PolynomialFeatures()
        {
            var pr = new MLNet.Models.Regression.PolyRegression();
            pr.GiveOptimizer(new Adam(0.05));
            pr.GiveLoss(new LSLoss {Constraint = Constraint.None});
            pr.GiveMetric(new MSE(), new MAE());
            pr.Fit(X, Y, new TrainConfig(2000));

            print(pr);
        }
    }
}