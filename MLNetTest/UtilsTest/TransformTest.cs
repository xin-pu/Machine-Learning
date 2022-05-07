using MLNet.Transforms;
using Numpy;
using Xunit;
using Xunit.Abstractions;

namespace MLNetTest.UtilsTest
{
    public class TransformTest : AbstractUnitTest
    {
        public TransformTest(ITestOutputHelper testOutputHelper)
            : base(testOutputHelper)
        {
            InputOneDim = np.array(new double[,] {{1}, {2}, {3}});
            InputMultiDim = np.array(new double[,] {{1, 1}, {2, 2}, {3, 3}});
        }

        protected NDarray InputOneDim { set; get; }
        protected NDarray InputMultiDim { set; get; }


        [Fact]
        public void LineFirstOrder()
        {
            var tran = new LinearFirstorder();
            var res = tran.Call(InputOneDim);
            print(res);
        }


        [Fact]
        public void Polynomial()
        {
            var tran = new Polynomial(2);
            var res = tran.Call(InputOneDim);
            print(res);
        }

        [Fact]
        public void TrianglePoly()
        {
            var tran = new TrianglePoly(2);
            var res = tran.Call(InputOneDim);
            print(res);
        }

        [Fact]
        public void PolyKernel()
        {
            var kernel = new Poly(1);
            var res = kernel.Call(InputMultiDim);
            print(res);
        }

        [Fact]
        public void GaussianKernel()
        {
            var kernel = new Gaussian(2);
            var res = kernel.Call(InputMultiDim);
            print(res);
        }

        [Fact]
        public void LaprasKernel()
        {
            var kernel = new Lapras(2);
            var res = kernel.Call(InputMultiDim);
            print(res);
        }

        [Fact]
        public void SigmoidKernel()
        {
            var kernel = new Sigmoid(2, -2);
            var res = kernel.Call(InputMultiDim);
            print(res);
        }
    }
}