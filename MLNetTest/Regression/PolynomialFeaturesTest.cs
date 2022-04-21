using MLNet;
using MLNet.Regression;
using Numpy;
using Xunit;
using Xunit.Abstractions;

namespace MLNetTest.Regression
{
    public class PolynomialFeaturesTest : AbstractUnitTest
    {
        public PolynomialFeaturesTest(ITestOutputHelper testOutputHelper)
            : base(testOutputHelper)
        {
            Log.print = print;
            var x = np.arange(-3, 3, 0.1);
            X = np.expand_dims(x, -1);
            Y = np.expand_dims(1 + 2 * np.sin(x), -1);
        }

        protected NDarray X { set; get; }
        protected NDarray Y { set; get; }


        [Fact]
        public void PolynomialFeatures()
        {
            var pf = new PolynomialFeatures(4)
            {
                Print = false
            };
            pf.Fit(X, Y, 1E-3, 10000);
            print(X);

            print(pf.Resolve);
            var evaluate = pf.Evaluate(X, Y);
            print(evaluate);

            print(Y);
            print(pf.Call(X));
        }

        [Fact]
        public void TrianglePolynomialFeatures()
        {
            var pf = new TrianglePolynomialFeatures(15)
            {
                Print = false
            };
            pf.Fit(X, Y, 1E-2, 1000);
            print(pf.Resolve);
            var evaluate = pf.Evaluate(X, Y);
            print(evaluate);

            print(Y);
            print(pf.Call(X));
        }

        [Fact]
        public void RidgePolynomialFeatures()
        {
            var ridge = new RigdePolynomialFeatures(4, 0.1);
            ridge.Fit(X, Y);
            ridge.PrintSelf();
            var evaluate = ridge.Evaluate(X, Y);
            print(evaluate);
        }
    }
}