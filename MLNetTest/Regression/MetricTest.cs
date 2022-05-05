using FluentAssertions;
using MLNet.LearningModel;
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
            var mse = Metric.getMSE(y_true, y_pred);
            mse.Should().Be(0.375);
        }

        [Fact]
        public void TestMAD()
        {
            var y_pred = np.array(2.5, 0.0, 2, 8);
            var y_true = np.array(3, -0.5, 2, 7);
            var mad = Metric.getMAD(y_true, y_pred);
            mad.Should().Be(0.5);
        }

        [Fact]
        public void TestEVS()
        {
            var y_true = np.array(3, -0.5, 2, 7);
            var y_pred = np.array(2.5, 0.0, 2, 8);
            var evs = Metric.getEVS(y_true, y_pred);
            print(evs);
            evs.Should().BeInRange(0.957, 0.958);
        }
    }
}