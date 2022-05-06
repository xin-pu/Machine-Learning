using MLNet.Kernels;
using MLNet.Metrics;
using MLNet.Model;
using MLNet.Model.Regression.LinearRegression;
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

            X = data[":,0:1"];
            Y = data[":,1:2"];
        }

        protected NDarray X { set; get; }
        protected NDarray Y { set; get; }


        [Fact]
        public void LinearRegression()
        {
            var lr = new MultipleLinearRegression
            {
                Print = true,
                Kernel = new Gaussian()
            };
            lr.Compile(null, null, new Metric[] {new MSE(), new EVS()});
            lr.Fit(X, Y, 0.1, 200);
            print(lr);
        }

        [Fact]
        public void TestSave()
        {
            var lr = new MultipleLinearRegression {Print = false};
            lr.Fit(X, Y, 0.1, 200);
            lr.Save("test.xml");

            var lrR = (MultipleLinearRegression) Model.Load("test.xml");
            print(lrR.Resolve);
        }
    }
}