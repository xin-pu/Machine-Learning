using MLNet.Optimizers;
using Numpy;
using Xunit;
using Xunit.Abstractions;

namespace MLNetTest.Regression
{
    public class OptimizerTest : AbstractUnitTest
    {
        public OptimizerTest(ITestOutputHelper testOutputHelper)
            : base(testOutputHelper)
        {
        }

        [Fact]
        public void TestSGD()
        {
            var weights = np.ones(2, 1);
            var grad = np.ones(2, 1);
            weights = new SGD(1E-2).Call(weights, grad);
            print(weights);
        }
    }
}