using System;
using System.Linq;
using MLNet.Utils;
using Xunit;
using Xunit.Abstractions;

namespace MLNetTest
{
    public class LRTest : AbstractUnitTest
    {
        public LRTest(ITestOutputHelper testOutputHelper)
            : base(testOutputHelper)
        {
        }


        private Func<double, double> func =>
            x => 2 * Math.Sin(2 * x) + Math.Pow(x, 3) + SystemRandomSource.NextDouble() * 0.02;


        [Fact]
        public void Solve()
        {
            var x = EnumerableExt.GetList(0, 3, 50);
            var y = x.ToList().Select(func).ToArray();

            //var lm = new LinearRegression(PrimaryType.Polynomial, 15);
            //lm.Fit(EnumerableExt.GetList(x), y);
            //if (lm.Theda != null) Print(lm.Theda);
        }
    }
}