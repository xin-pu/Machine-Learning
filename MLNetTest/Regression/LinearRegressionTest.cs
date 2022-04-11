using MLNet;
using MLNet.LearningModel;
using MLNet.Regression.LinearRegression;
using Numpy;
using Xunit;
using Xunit.Abstractions;

namespace MLNetTest.Regression
{
    public class LinearRegressionTest : AbstractUnitTest
    {
        private readonly double[,] x = {{1}, {2}, {3}, {4}};
        private readonly double[,] y = {{3}, {3.9}, {5}, {6.1}};

        public LinearRegressionTest(ITestOutputHelper testOutputHelper)
            : base(testOutputHelper)
        {
            Log.Print = Print;
        }


        [Fact]
        public void Solve()
        {
            var pf = new PolynomialFeatures(2) {SloveFunc = SloveFuc.Slove};
            pf.Fit(np.array(x), np.array(y));


            var x_val = np.array(new double[,] {{5}});
            var y_pred = pf.Predict(x_val);
        }
    }
}