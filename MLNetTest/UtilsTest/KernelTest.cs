using MLNet.Kernel;
using Numpy;
using Xunit;
using Xunit.Abstractions;

namespace MLNetTest.UtilsTest
{
    public class KernelTest : AbstractUnitTest
    {
        public KernelTest(ITestOutputHelper testOutputHelper)
            : base(testOutputHelper)
        {
            input = np.array(new double[,] {{1, 2}, {2, 4}, {3, 9}});
        }

        public NDarray input { set; get; }

        [Fact]
        public void PolyKernel()
        {
            var kernel = new PolyKernel(1);
            var res = kernel.Transform(input);
            print(res);
        }

        [Fact]
        public void GaussKernel()
        {
            var kernel = new GaussKernel(2);
            var res = kernel.Transform(input);
            print(res);
        }
    }
}