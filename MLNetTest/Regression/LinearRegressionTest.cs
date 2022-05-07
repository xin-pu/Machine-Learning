using System.Linq;
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
    public class LinearRegressionTest : AbstractUnitTest
    {
        private readonly string singledata = @"..\..\..\..\DataSet\data_singlevar.txt";


        public LinearRegressionTest(ITestOutputHelper testOutputHelper)
            : base(testOutputHelper)
        {
            var data = np2.load(singledata);

            //X = data[":,0:1"];
            //Y = data[":,1:2"];
            var x = Enumerable.Range(0, 100).Select(a => (double) a).ToArray();
            X = np.expand_dims(np.array(x), -1);
            Y = -2 + X * 1.5;
        }

        protected NDarray X { set; get; }
        protected NDarray Y { set; get; }


        [Fact]
        public void LinearRegression()
        {
            var lr = new MultipleLinearRegression();
            lr.GiveOptimizer(new SGD(1E-4));
            lr.GiveLoss(new LSLoss {Constraint = Constraint.None});
            lr.GiveMetric(new MSE(), new MAE());

            lr.Fit(X, Y, new TrainConfig(90));
            print(lr);
        }
    }
}