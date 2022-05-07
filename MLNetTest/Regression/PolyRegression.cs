using System.Linq;
using MLNet;
using MLNet.Losses;
using MLNet.Metrics;
using MLNet.Models.Regression;
using MLNet.Optimizers;
using Numpy;
using Xunit;
using Xunit.Abstractions;

namespace MLNetTest.Regression
{
    public class PolyRegression : AbstractUnitTest
    {
        public PolyRegression(ITestOutputHelper testOutputHelper)
            : base(testOutputHelper)
        {
            var x = Enumerable.Range(0, 100).Select(a => a * 0.05 - 1).ToArray();
            X = np.expand_dims(np.array(x), -1);
            Y = 1 - 3 * np.power(X, np.array(1)) + 2 * np.power(X, np.array(2));
        }

        protected NDarray X { set; get; }
        protected NDarray Y { set; get; }


        [Fact]
        public void PolynomialFeatures()
        {
            var pr = new MLNet.Models.Regression.PolyRegression(2);
            pr.GiveOptimizer(new SGD(2E-2));
            pr.GiveLoss(new LSLoss {Constraint = Constraint.L2});
            pr.GiveMetric(new MSE(), new MAE());
            pr.Fit(X, Y, new TrainConfig(500));

            print(pr);
        }


        //[Fact]
        //public void LassoPolynomialFeatures()
        //{
        //    var ridge = new PolynomialFeatures(4)
        //    {
        //        Constraint = Constraint.L1,
        //        Print = false
        //    };

        //    ridge.Fit(X, Y, 1E-3, 5000);
        //    ridge.PrintSelf();
        //}

        //[Fact]
        //public void RidgePolynomialFeatures()
        //{
        //    var ridge = new PolynomialFeatures(4)
        //    {
        //        Constraint = Constraint.L2,
        //        Print = false
        //    };

        //    ridge.Fit(X, Y, 1E-3, 5000);
        //    ridge.PrintSelf();
        //}

        //[Fact]
        //public void ElasticNetPolynomialFeatures()
        //{
        //    var ridge = new PolynomialFeatures(4)
        //    {
        //        Constraint = Constraint.ElasticNet,
        //        Print = false
        //    };

        //    ridge.Fit(X, Y, 1E-3, 5000);
        //    ridge.PrintSelf();
        //}


        //[Fact]
        //public void TrianglePolynomialFeatures()
        //{
        //    var pf = new TrianglePolynomialFeatures(15)
        //    {
        //        Constraint = Constraint.ElasticNet,
        //        Print = false
        //    };
        //    pf.Fit(X, Y, 1E-3, 5000);
        //    print(pf.Resolve);
        //    ;

        //    print(Y);
        //    print(pf.Call(X));
        //}
    }
}