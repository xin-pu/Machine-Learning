using MLNet;
using MLNet.Regression;
using MLNet.Regression.LinearRegression;
using Numpy;
using Xunit;
using Xunit.Abstractions;

namespace MLNetTest.Regression
{
    public class LinearRegressionTest : AbstractUnitTest
    {
        private readonly NDarray x = np.arange(-3, 3, 0.05);

        public LinearRegressionTest(ITestOutputHelper testOutputHelper)
            : base(testOutputHelper)
        {
            Log.Print = Print;
        }

        public NDarray X => np.expand_dims(x, -1);
        public NDarray Y => np.expand_dims(np.sin(x) + 0.3 * x, -1);

        [Fact]
        public void PolynomialFeatures()
        {
            var pf = new PolynomialFeatures(degree: 5)
            {
                SloveFunc = AbstractLinearRegression.SloveFuc.Slove
            };
            pf.Fit(X, Y);
            Print(Y);
            var y_pred = pf.Predict(X);
        }

        [Fact]
        public void Ridge()
        {
            var ridge = new Ridge(alpha: 0.1, degree: 5)
            {
                SloveFunc = AbstractLinearRegression.SloveFuc.Slove
            };
            ridge.Fit(X, Y);
            Print(Y);
            var y_pred = ridge.Predict(X);
        }


        [Fact]
        public void Lasso()
        {
            var lasso = new Lasso(alpha: 0.1, degree: 5)
            {
                SloveFunc = AbstractLinearRegression.SloveFuc.Slove
            };
            lasso.Fit(X, Y);
            Print(Y);
            var y_pred = lasso.Predict(X);
        }
    }
}