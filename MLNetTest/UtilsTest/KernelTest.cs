using MLNet.Kernels;
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
            input = np.array(new double[,] {{1, 1}, {2, 2}, {3, 3}});
        }

        public NDarray input { set; get; }

        [Fact]
        public void PolyKernel()
        {
            var kernel = new Poly(1);
            var res = kernel.Transform(input);
            print(res);
        }

        [Fact]
        public void GaussianKernel()
        {
            var kernel = new Gaussian(2);
            var res = kernel.Transform(input);
            print(res);
        }

        [Fact]
        public void LaprasKernel()
        {
            var kernel = new Lapras(2);
            var res = kernel.Transform(input);
            print(res);
        }

        [Fact]
        public void SigmoidKernel()
        {
            var kernel = new Sigmoid(2, -2);
            var res = kernel.Transform(input);
            print(res);
        }
    }
}