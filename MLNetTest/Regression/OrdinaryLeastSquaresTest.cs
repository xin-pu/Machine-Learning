using System;
using System.Linq;
using MLNet.Regression.LinearRegression;
using MLNet.Utils;
using Xunit;
using Xunit.Abstractions;

namespace MLNetTest.Regression
{
    public class OrdinaryLeastSquaresTest : AbstractUnitTest
    {
        public OrdinaryLeastSquaresTest(ITestOutputHelper testOutputHelper)
            : base(testOutputHelper)
        {
        }


        private Func<double, double> func =>
            x => Math.Sin(x + 1E-10) / (x + 1E-10) + 0.1 * x;


        [Fact]
        public void Solve()
        {
            var xLinear = EnumerableExt.GetLinearArray(-3, 3, 100);
            var yLinear = xLinear.ToList().Select(func).ToArray();
            Print(yLinear);
            var x = EnumerableExt.GetDyadicArray(xLinear);
            var y = EnumerableExt.GetDyadicArray(yLinear);


            var lm = new OrdinaryLeastSquares(PrimaryType.Polynomial, 15);
            lm.Slove(x, y);
            if (lm.Theda != null) Print(lm.Theda);

            var xLinear2 = EnumerableExt.GetLinearArray(-3, 3, 200);
            var x2 = EnumerableExt.GetDyadicArray(xLinear2);
            var y2 = lm.Predict(x2);
            Print(y2);
        }
    }
}