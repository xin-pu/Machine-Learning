using System;
using System.Linq;
using MLNet;
using MLNet.Regression;
using MLNet.Utils;
using Numpy;
using Xunit;
using Xunit.Abstractions;

namespace MLNetTest.Regression
{
    public class PolynomialFeaturesTest : AbstractUnitTest
    {
        //private readonly NDarray x = np.arange(-3, 3, 0.05);

        public PolynomialFeaturesTest(ITestOutputHelper testOutputHelper)
            : base(testOutputHelper)
        {
            Log.print = print;
            var x = np.arange(-3, 3, 0.1);
            X = np.expand_dims(x, -1);
            Y = np.expand_dims(1 + 2 * np2.power(x, 2) + 1.75 * np2.power(x, 3), -1);
        }

        protected NDarray X { set; get; }
        protected NDarray Y { set; get; }


        [Fact]
        public void PolynomialFeatures()
        {
            var pf = new PolynomialFeatures(degree: 4) {Print = true};
            pf.Fit(X, Y, 1E-3, 10000);
            print(pf.Resolve);
        }

        [Fact]
        public void Ridge()
        {
            var x = convert(X, 3);
            print(x);
            var res = SloveNone(x, Y);
            print(res);
        }

        /// <summary>
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <returns></returns>
        /// <exception cref="ArgumentException"></exception>
        public NDarray SloveNone(NDarray x, NDarray y)
        {
            var phiTranspose = np.transpose(x);
            var generalizedInverse = np.linalg.inv(np.matmul(phiTranspose, x));
            var res = np.matmul(np.matmul(generalizedInverse, phiTranspose), y);
            return res;
        }


        internal NDarray convert(NDarray x, int degree)
        {
            var batch = x.shape[0];
            var features = x.shape[1];
            if (features != 1) throw new Exception("Regression for 1 dims");

            var xTranspose = np.transpose(x);
            var npX = np.ones(degree + 1, batch);
            Enumerable.Range(1, degree).ToList().ForEach(d =>
            {
                var row = np.ones(x.shape[0]) * d;
                npX[d] = np.power(xTranspose, row);
            });
            npX = np.transpose(npX);
            return npX;
        }

        //[Fact]
        //public void Lasso()
        //{
        //    var lasso = new Lasso(alpha: 0.1, degree: 5)
        //    {
        //        SloveFunc = LinearRegressionModel.SloveFuc.Slove
        //    };
        //    lasso.Fit(X, Y);
        //    print(Y);
        //    var y_pred = lasso.Predict(X);
        //}
    }
}