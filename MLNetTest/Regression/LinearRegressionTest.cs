using MLNet;
using MLNet.Regression.LinearRegression;
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
            Log.print = print;
            var data = np2.load(singledata);

            X = data[":,0:1"];
            Y = data[":,1:2"];
        }

        protected NDarray X { set; get; }
        protected NDarray Y { set; get; }


        [Fact]
        public void Test()
        {
            var lr = new MultipleLinearRegression();
            lr.Fit(X, Y);
            print(lr.Resolve);
        }
    }
}