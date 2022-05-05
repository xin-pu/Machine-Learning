using System.Linq;
using FluentAssertions;
using MLNet.Metrics;
using Numpy;
using Xunit;
using Xunit.Abstractions;

namespace MLNetTest.Regression
{
    public class MetricTest : AbstractUnitTest
    {
        public MetricTest(ITestOutputHelper testOutputHelper) : base(testOutputHelper)
        {
        }

        [Fact]
        public void TestMSE()
        {
            var y_pred = np.array(2.5, 0.0, 2, 8);
            var y_true = np.array(3, -0.5, 2, 7);
            var mse = new MSE();
            mse.Call(y_true, y_pred).Should().Be(0.375);
            print(mse);
        }

        [Fact]
        public void TestMAD()
        {
            var y_pred = np.array(2.5, 0.0, 2, 8);
            var y_true = np.array(3, -0.5, 2, 7);
            var mad = new MAE();
            mad.Call(y_true, y_pred).Should().Be(0.5);
            print(mad);
        }

        [Fact]
        public void TestEVS()
        {
            var y_true = np.array(3, -0.5, 2, 7);
            var y_pred = np.array(2.5, 0.0, 2, 8);
            var evs = new EVS();
            evs.Call(y_true, y_pred).Should().BeInRange(0.957, 0.958);
            print(evs);
        }

        [Fact]
        public void TestRSquared()
        {
            var y_true = np.array(3, -0.5, 2, 7);
            var y_pred = np.array(2.5, 0.0, 2, 8);
            var r2 = new R2();
            r2.Call(y_true, y_pred).Should().BeInRange(0.948, 0.949);
            print(r2);
        }


        [Fact]
        public void TestAccuracy()
        {
            var y_true = np.array(1, 0, 1, 2, 2, 1, 1, 2, 1, 0);
            var y_pred = np.array(1, 1, 0, 1, 2, 0, 1, 2, 1, 0);
            var acc = new Accuracy();
            acc.Call(y_true, y_pred).Should().Be(0.6);
            print(acc);
        }

        [Fact]
        public void TestErrorRate()
        {
            var y_true = np.array(1, 0, 1, 2, 2, 1, 1, 2, 1, 0);
            var y_pred = np.array(1, 1, 0, 1, 2, 0, 1, 2, 1, 0);
            var errorRate = new ErrorRate();
            errorRate.Call(y_true, y_pred).Should().Be(0.4);
            print(errorRate);
        }

        [Fact]
        public void TestConfusionMatrixs()
        {
            var y_true = np.array(1, 0, 1, 2, 2, 1, 1, 2, 1, 0);
            var y_pred = np.array(1, 1, 0, 1, 2, 0, 1, 2, 1, 0);
            var confusionMatrixs = new ConfusionMatrixs(y_true, y_pred);
            foreach (var m in confusionMatrixs.ConfusionMatrixDict.Select(keyValuePair => keyValuePair.Value)) print(m);
        }
    }
}